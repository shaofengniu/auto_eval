from enum import Enum
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains.combine_documents.base import format_document
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.base import BasePromptTemplate

class DocumentProcessorType(str, Enum):
    CHARACTER_LIMIT_PROCESSOR = "character_limit_processor"
    DOCUMENT_COMPRESSOR = "document_compressor"


def _get_default_document_prompt() -> PromptTemplate:
    return PromptTemplate(input_variables=["page_content", "source", "title"],
                          template="Source:{source}\nTitle:{title}\n{page_content}")


class DocumentProcessor(BaseModel, ABC):
    document_prompt: BasePromptTemplate = Field(
        default_factory=_get_default_document_prompt
    )

    document_seperator: str = "\n" + "-" * 20 + "\n"

    def _combine_docs(self, docs: List[Document]) -> str:
        doc_strings = [format_document(doc, self.document_prompt) for doc in docs]
        return self.document_seperator.join(doc_strings)
    
    def process(self, docs: List[Document], **kwargs) -> str:
        return self._combine_docs(docs)
    

class CharacterLimitProcessor(DocumentProcessor):
    limit: int = 4000

    def process(self, docs: List[Document], query: str) -> str:
        return self._combine_docs(docs)[:self.limit]


class DocumentCompressor(DocumentProcessor):

    def process(self, documents: List[Document], query: str) -> str:
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
        compressor = LLMChainExtractor.from_llm(llm)
        compressed_docs = compressor.compress_documents(documents, query)
        return self._combine_docs(compressed_docs)


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