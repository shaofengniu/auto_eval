from pydantic import BaseModel
from typing import List
import json
from termcolor import colored
from langchain.document_loaders import ReadTheDocsLoader
from langchain.schema import Document, BaseRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.evaluation.qa import QAEvalChain
from retrievers import RetrieverType, initialize_retriever
from document_processors import (
    DocumentProcessorType,
    DocumentProcessor,
    initialize_document_processor,
)
from splitters import SplitterType, initialize_splitter
from models import ModelType, EmbeddingType, initialize_model, initialize_embedding
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
    document_processor_type: DocumentProcessorType = (
        DocumentProcessorType.CHARACTER_LIMIT_PROCESSOR
    )
    document_processor_args: dict = {}


class EvalInstance(BaseModel):
    retriever_type: RetrieverType
    retriever: BaseRetriever
    document_processor: DocumentProcessor

    class Config:
        arbitrary_types_allowed = True


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
    document_processor = initialize_document_processor(
        eval_conf.document_processor_type, **eval_conf.document_processor_args
    )
    print(eval_conf.dict(), len(raw_docs), len(docs))
    return EvalInstance(
        retriever=retriever,
        retriever_type=eval_conf.retriever_type,
        document_processor=document_processor,
    )


class DocumentLoaderType(str, Enum):
    READ_THE_DOCS = "read_the_docs"


DOCUMENT_LOADER_TYPE_TO_CLASS = {
    DocumentLoaderType.READ_THE_DOCS: ReadTheDocsLoader,
}


def load_docs(path: str) -> List[Document]:
    # TODO: support other document formats
    loader = ReadTheDocsLoader(path, features="html.parser")
    return loader.load()


template = """ 
Given the question: \n
{query}
Here are some documents retrieved by different retrievers in response to the question: \n
# Retriever <retriever_name>
## Document Content
<document_content>

# Retriever <retriever_name>
## Document Content
<document_content>

{result}
And here is the answer to the question: \n 
{answer}
Criteria: 
  relevance: Are the retrieved documents relevant to the question?"

Your response should be as follows:
# Retriever <retriever_name>
GRADE: (1 to 10, depending if the retrieved documents meet the criterion)
(line break)
JUSTIFICATION: (Write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Use one or two sentences maximum. Keep the answer as concise as possible.)
# Retriever <retriever_name>
GRADE: ...
JUSTIFICATION:...
"""

GRADE_DOCS_PROMPT = PromptTemplate(
    input_variables=["query", "answer", "result"], template=template
)


def grade_model_retrieval(examples, predictions, prompt):
    eval_chain = QAEvalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0),
        prompt=prompt,
        verbose=True,
    )
    outputs = eval_chain.evaluate(examples, predictions)
    return outputs


def build_qa_context(docs: List[Document], context_size: int) -> str:
    return "\n".join([doc.page_content for doc in docs])[:context_size]


def run_eval(evals, eval_qa_pair, grade_prompt):
    text = ""
    for eval in evals:
        docs = eval.retriever.get_relevant_documents(eval_qa_pair["query"])
        context = eval.document_processor.process(docs)
        text += f"# Retriever {eval.retriever_type}\n"
        text += "## Document Content\n"
        text += context + "\n\n"
    retrived_docs = [
        {
            "query": eval_qa_pair["query"],
            "answer": eval_qa_pair["answer"],
            "result": text,
        }
    ]
    graded_retrieval = grade_model_retrieval(
        [eval_qa_pair], retrived_docs, grade_prompt
    )
    return graded_retrieval[0], text


if __name__ == "__main__":
    init = True if len(sys.argv) > 1 and sys.argv[1] == "init" else False

    eval_confs = [
        EvalConfig(
            model_type=ModelType.CHAT_OPENAI,
            model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
            embedding_type=EmbeddingType.OPENAI_EMBEDDINGS,
            retriever_type=RetrieverType.LLAMA_DOC_SUMMARY,
            retriever_args={"init": init, "persist_path": "doc_summary_index"},
        ),
        EvalConfig(
            model_type=ModelType.CHAT_OPENAI,
            model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
            embedding_type=EmbeddingType.OPENAI_EMBEDDINGS,
            splitter_type=SplitterType.RECURSIVE_CHARACTER_TEXT_SPLITTER,
            splitter_args={"chunk_size": 1000, "chunk_overlap": 100},
            retriever_type=RetrieverType.CHROMA_VECTORSTORE,
            retriever_args={"init": init, "persist_path": "chroma_index"},
        ),
        EvalConfig(
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
        ),
    ]
    docs = load_docs(
        "python.langchain.com/en/latest/modules/indexes/retrievers/examples"
    )
    evals = [initialize_eval(e, docs) for e in eval_confs]
    qa_pairs = [{"query": "how to use hybrid retriever", "answer": "hybrid search"}]
    retrieved_documents = []
    for qa_pair in qa_pairs:
        graded_retrieval, retrieved_text = run_eval(evals, qa_pair, GRADE_DOCS_PROMPT)
        print(colored(graded_retrieval["text"], "cyan"))
        retrieved_documents.append(
            {"query": qa_pair["query"], "result": retrieved_text}
        )
        print(retrieved_text)
    with open("retrieved_documents.json", "w") as f:
        json.dump(retrieved_documents, f)
