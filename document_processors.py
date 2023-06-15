from enum import Enum
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document


class DocumentProcessorType(str, Enum):
    CHARACTER_LIMIT_PROCESSOR = "character_limit_processor"


class DocumentProcessor(BaseModel, ABC):
    @abstractmethod
    def process(self, documents: List[Document]) -> str:
        pass


class CharacterLimitProcessor(DocumentProcessor):
    limit: int = 2000

    def process(self, documents: List[Document]) -> List[Document]:
        return "\n".join(doc.page_content for doc in documents)[: self.limit]


DOCUMENT_PROCESSOR_TYPE_TO_CLASS = {
    DocumentProcessorType.CHARACTER_LIMIT_PROCESSOR: CharacterLimitProcessor,
}


def initialize_document_processor(
    document_processor_type: DocumentProcessorType, **kwargs
):
    if document_processor_type in DOCUMENT_PROCESSOR_TYPE_TO_CLASS:
        return DOCUMENT_PROCESSOR_TYPE_TO_CLASS[document_processor_type](
            **kwargs
        )
    raise ValueError(f"Unknown document processor type: {document_processor_type}")