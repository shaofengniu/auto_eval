from pydantic import BaseModel, Field
from typing import List, Tuple
from termcolor import colored
from langchain.document_loaders import (
    ReadTheDocsLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain.schema import Document, BaseRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.document_loaders.base import BaseLoader
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
from retrieval_qa import initialize_qa_chain
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
    qa_model_type: ModelType = ModelType.CHAT_OPENAI
    qa_model_args: dict = {}


class EvalInstance(BaseModel):
    config: EvalConfig
    retriever: BaseRetriever
    query_transformer: QueryTransformer
    document_processor: DocumentProcessor
    qa: LLMChain

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return f"{self.config.retriever_type}({self.config.query_transformer_type},{self.config.document_processor_type})"


def initialize_eval(eval_conf: EvalConfig, loader: BaseLoader) -> EvalInstance:
    llm = initialize_model(eval_conf.model_type, **eval_conf.model_args)
    embeddings = initialize_embedding(
        eval_conf.embedding_type, **eval_conf.embedding_args
    )

    text_splitter = initialize_splitter(
        eval_conf.splitter_type, eval_conf.splitter_args
    )

    retriever = initialize_retriever(
        llm, embeddings, loader, text_splitter, eval_conf.retriever_type, **eval_conf.retriever_args
    )
    query_transformer = initialize_query_transformer(
        llm, eval_conf.query_transformer_type, **eval_conf.query_transformer_args
    )
    document_processor = initialize_document_processor(
        eval_conf.document_processor_type, **eval_conf.document_processor_args
    )
    qa_llm = initialize_model(eval_conf.qa_model_type, **eval_conf.qa_model_args)
    qa_chain = initialize_qa_chain(qa_llm)

    print(eval_conf.dict())
    return EvalInstance(
        config=eval_conf,
        retriever=retriever,
        query_transformer=query_transformer,
        document_processor=document_processor,
        qa=qa_chain,
    )


class DocumentLoaderType(str, Enum):
    READ_THE_DOCS = "read_the_docs"
    DIRECTORY = "directory"


DOCUMENT_LOADER_TYPE_TO_CLASS = {
    DocumentLoaderType.READ_THE_DOCS: ReadTheDocsLoader,
    DocumentLoaderType.DIRECTORY: DirectoryLoader,
}


def initialize_document_loader(path: str, loader_type: DocumentLoaderType, **kwargs) -> BaseLoader:
    # TODO: support other document formats
    if loader_type not in DOCUMENT_LOADER_TYPE_TO_CLASS:
        raise ValueError(f"Unknown document loader type: {loader_type}")
    loader = DOCUMENT_LOADER_TYPE_TO_CLASS[loader_type](path, **kwargs)
    return loader


template_2 = """
You are a helpful and precise assistant for checking the quality of the documents retrieved by two AI assistants.

[Question]
{question}

[The Start of Assistant 1's Documents]
{documents_1}

[The End of Assistant 1's Documents]

[The Start of Assistant 2's Documents]
{documents_2}

[The End of Assistant 2's Documents]

We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their documents. 
Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
{format_instructions}
"""


class CompareResult(BaseModel):
    score_1: int = Field(description="Score for Assistant 1")
    score_2: int = Field(description="Score for Assistant 2")
    explanation: str = Field(
        "A comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
    )

    def __str__(self) -> str:
        return (
            f"* Assistant 1: {self.score_1}\n"
            f"* Assistant 2: {self.score_2}\n"
            f"* Explanation: {self.explanation}"
        )


output_parser = PydanticOutputParser(pydantic_object=CompareResult)

EVAL_PROMPT = PromptTemplate(
    input_variables=["question", "documents_1", "documents_2"],
    template=template_2,
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)


def compare_model_retrieval(
    question: str, documents_1: str, documents_2: str
) -> CompareResult:
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    eval_chain = LLMChain(llm=llm, prompt=EVAL_PROMPT, output_parser=output_parser)
    output = eval_chain.predict(
        question=question, documents_1=documents_1, documents_2=documents_2
    )
    return output


def build_qa_context(docs: List[Document], context_size: int) -> str:
    sep = "\n" + "-" * 20 + "\n"
    return sep.join([str(doc.metadata) + "\n" + doc.page_content for doc in docs])[:context_size]


def run_and_compare(evals: Tuple[EvalInstance, EvalInstance], question: str):
    queries = []
    retrieved_texts = []
    answers = []
    for eval in evals:
        query = eval.query_transformer.transform(question)
        docs = eval.retriever.get_relevant_documents(query)
        text = eval.document_processor.process(docs, question)
        answer = eval.qa.run({"question": question, "context": text})
        queries.append(query)
        retrieved_texts.append(text)
        answers.append(answer)
    result = compare_model_retrieval(question, retrieved_texts[0], retrieved_texts[1])
    return result, queries, retrieved_texts, answers


def run_evals(evals):
    questions = [
        "How do I extract information like token usage from the LLMResult in LangChain?",
        "What is LLMChain in LangChain and how do I use it with chat models?",
        "How do I use agents with chat models in LangChain?",
        "How do I load tools to use with LangChain and how do they work?",
    ]

    results = []
    for question in questions:
        print(colored("* Question: " + question, "red"))
        result, queries, retrieved_texts, answers = run_and_compare(evals, question)
        print(colored(str(result), "green"))
        results.append(
            {
                "question": question,
                "queries": queries,
                "retrieved_texts": retrieved_texts,
                "answers": answers,
                "result": result,
            }
        )
    total_score_1 = sum(r["result"].score_1 for r in results)
    total_score_2 = sum(r["result"].score_2 for r in results)
    print(f"Total Score for Assistant 1: {total_score_1}")
    print(f"Total Score for Assistant 2: {total_score_2}")
    with open("results.md", "w") as f:
        for i, result in enumerate(results):
            f.write(f"# Question {i+1}\n\n" + result["question"] + "\n\n")
            f.write("## Result\n\n" + str(result["result"]) + "\n\n")
            f.write("## Answers\n\n")
            f.write("### Answer 1\n\n```\n\n"
                    + result["answers"][0].replace("```", "")
                    + "\n\n```\n\n")
            f.write("### Answer 2\n\n```\n\n" 
                    + result["answers"][1].replace("```", "")
                    + "\n\n```\n\n")
            f.write("## Documents\n\n")
            f.write(
                "### Documents 1\n\n"
                + result["queries"][0] + "\n\n"
                + "```\n\n"
                + result["retrieved_texts"][0].replace("```", "")
                + "\n\n```\n\n"
            )
            f.write(
                "### Documents 2\n\n"
                + result["queries"][1] + "\n\n"
                + "```\n\n"
                + result["retrieved_texts"][1].replace("```", "")
                + "\n\n```\n\n"
            )




if __name__ == "__main__":
    init = True if len(sys.argv) > 1 and sys.argv[1] == "init" else False
    eval_chroma = EvalConfig(
        model_type=ModelType.CHAT_OPENAI,
        model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
        embedding_type=EmbeddingType.OPENAI_EMBEDDINGS,
        splitter_type=SplitterType.RECURSIVE_CHARACTER_TEXT_SPLITTER,
        splitter_args={"chunk_size": 1000, "chunk_overlap": 100},
        retriever_type=RetrieverType.CHROMA_VECTORSTORE,
        retriever_args={"init": init, "persist_path": "index/chroma_index", "search_kwargs": {"k": 10}},
        document_processor_type=DocumentProcessorType.CHARACTER_LIMIT_PROCESSOR,
        document_processor_args={"limit": 4000},
        qa_model_type=ModelType.CHAT_OPENAI,
        qa_model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
    )
    eval_chroma_non_split = EvalConfig(
        model_type=ModelType.CHAT_OPENAI,
        model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
        embedding_type=EmbeddingType.OPENAI_EMBEDDINGS,
        splitter_type=SplitterType.DEFAULT_TEXT_SPLITTER,
        retriever_type=RetrieverType.CHROMA_VECTORSTORE,
        retriever_args={"init": init, "persist_path": "index/chroma_index_non_split"},
        document_processor_type=DocumentProcessorType.CHARACTER_LIMIT_PROCESSOR,
        document_processor_args={"limit": 4000},
        qa_model_type=ModelType.CHAT_OPENAI,
        qa_model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
    )
    eval_chroma_keywords = EvalConfig(
        model_type=ModelType.CHAT_OPENAI,
        model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
        embedding_type=EmbeddingType.OPENAI_EMBEDDINGS,
        splitter_type=SplitterType.RECURSIVE_CHARACTER_TEXT_SPLITTER,
        splitter_args={"chunk_size": 1000, "chunk_overlap": 100},
        retriever_type=RetrieverType.CHROMA_VECTORSTORE,
        retriever_args={"init": init, "persist_path": "index/chroma_index"},
        query_transformer_type=QueryTransformerType.KEYWORDS_QUERY_TRANSFORMER,
        document_processor_type=DocumentProcessorType.CHARACTER_LIMIT_PROCESSOR,
        document_processor_args={"limit": 4000},
        qa_model_type=ModelType.CHAT_OPENAI,
        qa_model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
    )
    eval_chroma_with_compressor = EvalConfig(
        model_type=ModelType.CHAT_OPENAI,
        model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
        embedding_type=EmbeddingType.OPENAI_EMBEDDINGS,
        splitter_type=SplitterType.RECURSIVE_CHARACTER_TEXT_SPLITTER,
        splitter_args={"chunk_size": 2000, "chunk_overlap": 100},
        retriever_type=RetrieverType.CHROMA_VECTORSTORE,
        retriever_args={
            "init": init,
            "persist_path": "index/chroma_index_with_compressor",
        },
        document_processor_type=DocumentProcessorType.DOCUMENT_COMPRESSOR,
        document_processor_args={"limit": 4000},
        qa_model_type=ModelType.CHAT_OPENAI,
        qa_model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
    )
    eval_llama_doc_summary = EvalConfig(
        model_type=ModelType.CHAT_OPENAI,
        model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
        embedding_type=EmbeddingType.OPENAI_EMBEDDINGS,
        retriever_type=RetrieverType.LLAMA_DOC_SUMMARY,
        retriever_args={"init": init, "persist_path": "doc_summary_index"},
        qa_model_type=ModelType.CHAT_OPENAI,
        qa_model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
    )
    eval_hybrid_search = EvalConfig(
        model_type=ModelType.CHAT_OPENAI,
        model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
        embedding_type=EmbeddingType.OPENAI_EMBEDDINGS,
        splitter_type=SplitterType.RECURSIVE_CHARACTER_TEXT_SPLITTER,
        splitter_args={"chunk_size": 1000, "chunk_overlap": 100},
        query_transformer_type=QueryTransformerType.KEYWORDS_QUERY_TRANSFORMER,
        retriever_type=RetrieverType.WEAVIATE_HYBRID_SEARCH,
        retriever_args={
            "init": init,
            "persist_path": "hybrid_index",
            "url": "http://localhost:8080",
            "index_name": "LangChain",
            "text_key": "text",
        },
        qa_model_type=ModelType.CHAT_OPENAI,
        qa_model_args={"model_name": "gpt-3.5-turbo-0613", "temperature": 0},
    )

    eval_confs = [eval_chroma]
    loader = initialize_document_loader(
        "python.langchain.com/en/latest/",
        DocumentLoaderType.READ_THE_DOCS,
        features="lxml",
    )
    evals = [initialize_eval(e, loader) for e in eval_confs]
    if len(evals) == 2:
        run_evals(evals)
    else:
        while True:
            retriever = evals[0].retriever
            query = input(colored("Enter question ===> ", "red"))
            docs = retriever.get_relevant_documents(query)
            for doc in docs:
                print(colored("#" * 80, "cyan"))
                if "title" in doc.metadata:
                    print(colored("* " + doc.metadata["title"], "green"))
                if "source" in doc.metadata:
                    print(colored("* " + doc.metadata["source"], "yellow"))
                if "id" in doc.metadata:
                    print(colored("* " + doc.metadata["id"], "magenta"))
                print(doc.page_content.replace(query, colored("#" + query + "#", "red")))