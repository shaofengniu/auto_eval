from enum import Enum
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


class SplitterType(str, Enum):
    RECURSIVE_CHARACTER_TEXT_SPLITTER = "recursive_character_text_splitter"
    DEFAULT_TEXT_SPLITTER = "default_text_splitter"


class DefaultTextSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        return [text]


SPLITTER_TYPE_TO_CLASS = {
    SplitterType.RECURSIVE_CHARACTER_TEXT_SPLITTER: RecursiveCharacterTextSplitter,
    SplitterType.DEFAULT_TEXT_SPLITTER: DefaultTextSplitter
}


def initialize_splitter(splitter_type: SplitterType, kwargs) -> TextSplitter:
    if splitter_type in SPLITTER_TYPE_TO_CLASS:
        return SPLITTER_TYPE_TO_CLASS[splitter_type](**kwargs)
    else:
        raise NotImplementedError(f"Splitter {splitter_type} not implemented")
