from enum import Enum
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Optional
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.llm import LLMChain

class PostProcessorType(str, Enum):
    DOCUMENT_COMPRESSOR = "document_compressor"
    LLM_RERANKER = "llm_reranker"


class PostProcessor(BaseModel, ABC):
    @abstractmethod
    def process(self, docs: List[Document], query: str) -> List[Document]:
        pass
    

class DocumentCompressor(PostProcessor):

    def process(self, documents: List[Document], query: str) -> str:
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
        compressor = LLMChainExtractor.from_llm(llm)
        compressed_docs = compressor.compress_documents(documents, query)
        return compressed_docs


DEFAULT_DOCUMENT_RERANK_PROMPT_TMPL = (
    "A list of documents is shown below. Each document has a number next to it along "
    "with the content of the document. A question is also provided. \n"
    "Respond with the numbers of the documents "
    "you should consult to answer the question, in order of relevance, as well \n"
    "as the relevance score. The relevance score is a number from 1-10 based on "
    "how relevant you think the document is to the question.\n"
    "Do not include any documents that are not relevant to the question. \n"
    "Example format: \n"
    "Document 1:\n<content of document 1>\n\n"
    "Document 2:\n<content of document 2>\n\n"
    "...\n\n"
    "Document 10:\n<content of document 10>\n\n"
    "Question: <question>\n"
    "Answer:\n"
    "Doc: 9, Relevance: 7\n"
    "Doc: 3, Relevance: 4\n"
    "Doc: 7, Relevance: 3\n\n"
    "Let's try this now: \n\n"
    "{context}\n"
    "Question: {query}\n"
    "Answer:\n"
)


DEFAULT_DOCUMENT_RERANK_PROMPT = PromptTemplate(
    input_variables=["context", "query"],
    template=DEFAULT_DOCUMENT_RERANK_PROMPT_TMPL
)

class LLMReranker(PostProcessor):

    rerank_prompt = DEFAULT_DOCUMENT_RERANK_PROMPT
    batch_size: int = 4
    k: int = 4

    def process(self, docs: List[Document], query: str) -> List[Document]:
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
        rerank_chain = LLMChain(llm=llm, prompt=self.rerank_prompt)
        reranked_docs = []
        for idx in range(0, len(docs), self.batch_size):
            docs_to_rerank = docs[idx:idx+self.batch_size]
            context = self._format_context(docs_to_rerank)
            raw_response = rerank_chain.predict(context=context, query=query)
            doc_nums_and_scores = self._parse_response(raw_response)
            docs_and_scores = [(docs_to_rerank[num-1], score) for num, score in doc_nums_and_scores]
            reranked_docs.extend(docs_and_scores)
        reranked_docs = sorted(reranked_docs, key=lambda x: x[1], reverse=True)
        reranked_docs = [t[0] for t in reranked_docs]
        return reranked_docs[:self.k]
            
    
    def _parse_response(self, response: str) -> List[Tuple[int, Optional[float]]]:
        lines = response.split("\n")
        answers = []
        for line in lines:
            line_tokens = line.split(",")
            if len(line_tokens) != 2:
                continue
            answer_num = int(line_tokens[0].split(":")[1].strip())
            score = float(line_tokens[1].split(":")[1].strip())
            answers.append((answer_num, score))
        return answers


    def _format_context(self, docs: List[Document]) -> str:
        context = []
        for index, doc in enumerate(docs):
            num = index + 1
            context.append(f"Document {num}:\n{doc.page_content}")
        return "\n\n".join(context)

        


DOCUMENT_PROCESSOR_TYPE_TO_CLASS = {
    PostProcessorType.DOCUMENT_COMPRESSOR: DocumentCompressor,
    PostProcessorType.LLM_RERANKER: LLMReranker
}


def initialize_post_processor(
    document_processor_type: PostProcessorType, **kwargs
):
    if document_processor_type in DOCUMENT_PROCESSOR_TYPE_TO_CLASS:
        return DOCUMENT_PROCESSOR_TYPE_TO_CLASS[document_processor_type](
            **kwargs
        )
    raise ValueError(f"Unknown document processor type: {document_processor_type}")