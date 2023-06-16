from abc import ABC
from typing import List
from langchain.schema import Document, BaseRetriever
from llama import LlamaRetriever
from llama_index import (
    Document as LlamaDocument,
    LLMPredictor,
    ResponseSynthesizer,
    ServiceContext,
    StorageContext,
)
from llama_index.indices.document_summary import (
    DocumentSummaryIndex,
    DocumentSummaryIndexRetriever,
)
from llama_index.indices.loading import load_index_from_storage
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.vectorstores import Chroma, Weaviate
from langchain.base_language import BaseLanguageModel
from enum import Enum
import weaviate
import os


class RetrieverBuilder(ABC):
    @staticmethod
    def from_documents(
        llm, embeddings, docs: List[Document], **kwargs
    ) -> BaseRetriever:
        pass


class RetrieverType(str, Enum):
    LLAMA_DOC_SUMMARY = "llama_doc_summary"
    CHROMA_VECTORSTORE = "chroma_vectorstore"
    WEAVIATE_HYBRID_SEARCH = "weaviate_hybrid_search"


class ChromaRetrieverBuilder(RetrieverBuilder):
    @staticmethod
    def from_documents(
        llm, embeddings, docs: List[Document], **kwargs
    ) -> BaseRetriever:
        if (
            not kwargs["init"]
            and "persiste_path" in kwargs
            and os.path.exists(kwargs["persiste_path"])
        ):
            return Chroma(
                persist_directory=kwargs["persist_path"], embedding_function=embeddings
            ).as_retriever()
        else:
            return Chroma.from_documents(
                docs, embeddings, persist_directory=kwargs["persist_path"]
            ).as_retriever()


class WeaviateHybridSearchRetrieverBuilder(RetrieverBuilder):
    @staticmethod
    def from_documents(
        llm, embeddings, docs: List[Document], **kwargs
    ) -> BaseRetriever:
        if kwargs["init"]:
            Weaviate.from_documents(
                docs,
                embeddings,
                by_text=False,
                weaviate_url=kwargs["url"],
                index_name=kwargs["index_name"],
                text_key=kwargs["text_key"],
            )
        client = weaviate.Client(url=kwargs["url"])
        return WeaviateHybridSearchRetriever(
            client, kwargs["index_name"], kwargs["text_key"], embeddings
        )


class LlamaDocSummaryRetrieverBuilder(RetrieverBuilder):
    @staticmethod
    def _from_documents(
        llm: BaseLanguageModel, embeddings, docs: List[Document], **kwargs
    ) -> "LlamaRetriever":
        service_context = ServiceContext.from_defaults(
            llm_predictor=LLMPredictor(llm), chunk_size=1024
        )
        response_synthesizer = ResponseSynthesizer.from_args(
            response_mode="tree_summarize"
        )
        docs = [LlamaDocument.from_langchain_format(d) for d in docs]
        doc_summary_index = DocumentSummaryIndex.from_documents(
            docs,
            service_context=service_context,
            response_synthesizer=response_synthesizer,
        )

        doc_summary_retriever = DocumentSummaryIndexRetriever(doc_summary_index)

        retriever = LlamaRetriever(
            index=doc_summary_index, retriever=doc_summary_retriever
        )
        retriever.persist(kwargs["persist_path"])
        return retriever

    @staticmethod
    def _from_storage(llm: BaseLanguageModel, embeddings, **kwargs) -> "LlamaRetriever":
        storage_context = StorageContext.from_defaults(
            persist_dir=kwargs["persist_path"]
        )
        doc_summary_index = load_index_from_storage(storage_context)
        service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm))
        doc_summary_retriever = DocumentSummaryIndexRetriever(
            doc_summary_index, service_context=service_context
        )
        return LlamaRetriever(index=doc_summary_index, retriever=doc_summary_retriever)

    @staticmethod
    def from_documents(
        llm, embeddings, docs: List[Document], **kwargs
    ) -> BaseRetriever:
        if (
            not kwargs["init"]
            and "persist_path" in kwargs
            and os.path.exists(kwargs["persist_path"])
        ):
            return LlamaDocSummaryRetrieverBuilder._from_storage(
                llm, embeddings, **kwargs
            )
        else:
            return LlamaDocSummaryRetrieverBuilder._from_documents(
                llm, embeddings, docs, **kwargs
            )


RETRIEVER_TYPE_TO_CLASS = {
    RetrieverType.LLAMA_DOC_SUMMARY: LlamaDocSummaryRetrieverBuilder,
    RetrieverType.CHROMA_VECTORSTORE: ChromaRetrieverBuilder,
    RetrieverType.WEAVIATE_HYBRID_SEARCH: WeaviateHybridSearchRetrieverBuilder,
}


def initialize_retriever(
    llm, embeddings, docs: List[Document], retriever_type: str, **kwargs
) -> BaseRetriever:
    return RETRIEVER_TYPE_TO_CLASS[retriever_type].from_documents(
        llm, embeddings, docs, **kwargs
    )
