from enum import Enum
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.document_compressors import LLMChainExtractor

class DocumentProcessorType(str, Enum):
    CHARACTER_LIMIT_PROCESSOR = "character_limit_processor"
    DOCUMENT_COMPRESSOR = "document_compressor"


class DocumentProcessor(BaseModel, ABC):
    @abstractmethod
    def process(self, documents: List[Document], query: str) -> str:
        pass


class CharacterLimitProcessor(DocumentProcessor):
    limit: int = 2000

    def process(self, documents: List[Document], query: str) -> List[Document]:
        return "\n".join(doc.page_content for doc in documents)[: self.limit]


class DocumentCompressor(DocumentProcessor):
    limit: int = 2000

    def process(self, documents: List[Document], query: str) -> str:
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
        compressor = LLMChainExtractor.from_llm(llm)
        compressed_docs = compressor.compress_documents(documents, query)
        return "\n".join(doc.page_content for doc in compressed_docs)[: self.limit]


DOCUMENT_PROCESSOR_TYPE_TO_CLASS = {
    DocumentProcessorType.CHARACTER_LIMIT_PROCESSOR: CharacterLimitProcessor,
    DocumentProcessorType.DOCUMENT_COMPRESSOR: DocumentCompressor,
}


def initialize_document_processor(
    document_processor_type: DocumentProcessorType, **kwargs
):
    if document_processor_type in DOCUMENT_PROCESSOR_TYPE_TO_CLASS:
        return DOCUMENT_PROCESSOR_TYPE_TO_CLASS[document_processor_type](
            **kwargs
        )
    raise ValueError(f"Unknown document processor type: {document_processor_type}")