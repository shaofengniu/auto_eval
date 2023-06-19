from pydantic import BaseModel, Field
from typing import List
import json
from termcolor import colored
from langchain.document_loaders import ReadTheDocsLoader
from langchain.schema import Document, BaseRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.evaluation.qa import QAEvalChain
from langchain.chains.llm import LLMChain
from retrievers import RetrieverType, initialize_retriever
from document_processors import (
    DocumentProcessorType,
    DocumentProcessor,
    initialize_document_processor,
)
from splitters import SplitterType, initialize_splitter
from models import ModelType, EmbeddingType, initialize_model, initialize_embedding
from query_transformers import (
    QueryTransformerType,
    QueryTransformer,
    initialize_query_transformer,
)
from enum import Enum
import sys


class EvalConfig(BaseModel):
    model_type: ModelType = ModelType.CHAT_OPENAI
    model_args: dict = {}
    embedding_type: EmbeddingType = EmbeddingType.OPENAI_EMBEDDINGS
    embedding_args: dict = {}
    splitter_type: SplitterType = SplitterType.DEFAULT_TEXT_SPLITTER
    splitter_args: dict = {}
    retriever_type: RetrieverType
    retriever_args: dict = {}
    query_transformer_type: QueryTransformerType = (
        QueryTransformerType.DEFAULT_QUERY_TRANSFORMER
    )
    query_transformer_args: dict = {}
    document_processor_type: DocumentProcessorType = (
        DocumentProcessorType.CHARACTER_LIMIT_PROCESSOR
    )
    document_processor_args: dict = {}


class EvalInstance(BaseModel):
    config: EvalConfig
    retriever: BaseRetriever
    query_transformer: QueryTransformer
    document_processor: DocumentProcessor

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return f"{self.config.retriever_type}({self.config.document_processor_type})"


def initialize_eval(eval_conf: EvalConfig, raw_docs: List[Document]):
    llm = initialize_model(eval_conf.model_type, **eval_conf.model_args)
    embeddings = initialize_embedding(
        eval_conf.embedding_type, **eval_conf.embedding_args
    )

    text_splitter = initialize_splitter(
        eval_conf.splitter_type, eval_conf.splitter_args
    )
    docs = text_splitter.split_documents(raw_docs)
    retriever = initialize_retriever(
        llm, embeddings, docs, eval_conf.retriever_type, **eval_conf.retriever_args
    )
    query_transformer = initialize_query_transformer(
        eval_conf.query_transformer_type, **eval_conf.query_transformer_args
    )
    document_processor = initialize_document_processor(
        eval_conf.document_processor_type, **eval_conf.document_processor_args
    )
    print(eval_conf.dict(), len(raw_docs), len(docs))
    return EvalInstance(
        config=eval_conf,
        retriever=retriever,
        query_transformer=query_transformer,
        document_processor=document_processor,
    )


class DocumentLoaderType(str, Enum):
    READ_THE_DOCS = "read_the_docs"


DOCUMENT_LOADER_TYPE_TO_CLASS = {
    DocumentLoaderType.READ_THE_DOCS: ReadTheDocsLoader,
}


def load_docs(path: str, loader_type: DocumentLoaderType, **kwargs) -> List[Document]:
    # TODO: support other document formats
    if loader_type not in DOCUMENT_LOADER_TYPE_TO_CLASS:
        raise ValueError(f"Unknown document loader type: {loader_type}")
    loader = DOCUMENT_LOADER_TYPE_TO_CLASS[loader_type](path, **kwargs)
    return loader.load()


template = """ 
Given the question: \n
{query}

And here is the answer to the question: \n 
{answer}

Here are some documents retrieved in response to the question: \n
{text}

Criteria: 
  relevance: Are the retrieved documents relevant to the question?"

{format_instructions}
"""


t = """
Your response should be as follows:
# <retriever_name>
GRADE: (1 to 10, depending if the retrieved documents meet the criterion)
(line break)
JUSTIFICATION: (Write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Use one or two sentences maximum. Keep the answer as concise as possible.)
# <retriever_name>
GRADE: ...
JUSTIFICATION:...
"""


from langchain.output_parsers import PydanticOutputParser


class EvalResult(BaseModel):
    grade: int = Field(
        description="1 to 10, depending if the retrieved documents meet the criterion"
    )
    justification: str = Field(
        description="Write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Use one or two sentences maximum. Keep the answer as concise as possible."
    )


parser = PydanticOutputParser(pydantic_object=EvalResult)
GRADE_DOCS_PROMPT = PromptTemplate(
    input_variables=["query", "answer", "text"],
    template=template,
    output_parser=parser,
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def eval_model_retrieval(query: str, answer: str, text: str) -> EvalResult:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
    eval_chain = LLMChain(llm=llm, prompt=GRADE_DOCS_PROMPT)
    output = eval_chain.predict_and_parse(query=query, answer=answer, text=text)
    return output


def build_qa_context(docs: List[Document], context_size: int) -> str:
    return "\n".join([doc.page_content for doc in docs])[:context_size]


def run_eval(evals: List[EvalInstance], eval_qa_pair):
    eval_results = {}
    retrieved_text = {}
    for eval in evals:
        query = eval.query_transformer.transform(eval_qa_pair["query"])
        docs = eval.retriever.get_relevant_documents(query)
        text = eval.document_processor.process(docs, eval_qa_pair["query"])
        eval_result = eval_model_retrieval(
            eval_qa_pair["query"], eval_qa_pair["answer"], text
        )
        eval_results[str(eval)] = eval_result
        retrieved_text[str(eval)] = text
    return eval_results, retrieved_text


if __name__ == "__main__":
    init = True if len(sys.argv) > 1 and sys.argv[1] == "init" else False
    eval_chroma = EvalConfig(
        model_type=ModelType.CHAT_OPENAI,
        model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
        embedding_type=EmbeddingType.OPENAI_EMBEDDINGS,
        splitter_type=SplitterType.RECURSIVE_CHARACTER_TEXT_SPLITTER,
        splitter_args={"chunk_size": 1000, "chunk_overlap": 100},
        retriever_type=RetrieverType.CHROMA_VECTORSTORE,
        retriever_args={"init": init, "persist_path": "chroma_index"},
        document_processor_type=DocumentProcessorType.CHARACTER_LIMIT_PROCESSOR,
        document_processor_args={"limit": 4000},
    )
    eval_chroma_with_compressor = EvalConfig(
        model_type=ModelType.CHAT_OPENAI,
        model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
        embedding_type=EmbeddingType.OPENAI_EMBEDDINGS,
        splitter_type=SplitterType.RECURSIVE_CHARACTER_TEXT_SPLITTER,
        splitter_args={"chunk_size": 2000, "chunk_overlap": 100},
        retriever_type=RetrieverType.CHROMA_VECTORSTORE,
        retriever_args={"init": init, "persist_path": "chroma_index_with_compressor"},
        document_processor_type=DocumentProcessorType.DOCUMENT_COMPRESSOR,
        document_processor_args={"limit": 4000},
    )
    eval_llama_doc_summary = EvalConfig(
        model_type=ModelType.CHAT_OPENAI,
        model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
        embedding_type=EmbeddingType.OPENAI_EMBEDDINGS,
        retriever_type=RetrieverType.LLAMA_DOC_SUMMARY,
        retriever_args={"init": init, "persist_path": "doc_summary_index"},
    )
    eval_hybrid_search = EvalConfig(
        model_type=ModelType.CHAT_OPENAI,
        model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
        embedding_type=EmbeddingType.OPENAI_EMBEDDINGS,
        splitter_type=SplitterType.RECURSIVE_CHARACTER_TEXT_SPLITTER,
        splitter_args={"chunk_size": 1000, "chunk_overlap": 100},
        retriever_type=RetrieverType.WEAVIATE_HYBRID_SEARCH,
        retriever_args={
            "init": init,
            "persist_path": "hybrid_index",
            "url": "http://localhost:8080",
            "index_name": "LangChain_4434c0821b20463b878724ede4b28322",
            "text_key": "text",
        },
    )

    eval_confs = [eval_chroma,
                  eval_chroma_with_compressor,
                  eval_llama_doc_summary,
                  eval_hybrid_search]
    docs = load_docs(
        "python.langchain.com/en/latest/modules/indexes/retrievers/examples",
        DocumentLoaderType.READ_THE_DOCS,
        features="html.parser",
    )
    evals = [initialize_eval(e, docs) for e in eval_confs]
    qa_pairs = [{"query": "how to use hybrid retriever", "answer": "hybrid search"}]
    retrieved_documents = []
    results = []
    for qa_pair in qa_pairs:
        eval_results, retrieved_text = run_eval(evals, qa_pair)
        print("QUERY: ", qa_pair["query"])
        for k, v in eval_results.items():
            print(colored(k, "blue"), colored(v, "green"))
            results.append({
                "query": qa_pair["query"],
                "retriever": k,
                "grade": v.grade,
                "justification": v.justification,
                "retrieved_text": retrieved_text[k]
            })
    with open("results.txt", "w") as f:
        for result in results:
            f.write(f"QUERY: {result['query']}\n")
            f.write(f"RETRIEVER: {result['retriever']}\n")
            f.write(f"GRADE: {result['grade']}\n")
            f.write(f"JUSTIFICATION: {result['justification']}\n")
            f.write(f"RETRIEVED TEXT: {result['retrieved_text']}\n\n")
