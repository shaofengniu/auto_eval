from enum import Enum
from pydantic import BaseModel
from abc import ABC, abstractmethod
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.hyde.prompts import PROMPT_MAP


class QueryTransformerType(str, Enum):
    DEFAULT_QUERY_TRANSFORMER = "default_transformer"
    HYDE_QUERY_TRANSFORMER = "hyde_query_transformer"


class QueryTransformer(BaseModel, ABC):
    @abstractmethod
    def transform(self, query: str) -> str:
        pass


class DefaultQueryTransformer(QueryTransformer):
    def transform(self, query: str) -> str:
        return query
    

class HydeQueryTransformer(QueryTransformer):
    def transform(self, query: str) -> str:
        raise NotImplementedError("HydeQueryTransformer not implemented")


QUERY_TRANSFORMER_TYPE_TO_CLASS = {
    QueryTransformerType.DEFAULT_QUERY_TRANSFORMER: DefaultQueryTransformer,
    QueryTransformerType.HYDE_QUERY_TRANSFORMER: HydeQueryTransformer
}


def initialize_query_transformer(
    query_transformer_type: QueryTransformerType, **kwargs
) -> QueryTransformer:
    if query_transformer_type in QUERY_TRANSFORMER_TYPE_TO_CLASS:
        return QUERY_TRANSFORMER_TYPE_TO_CLASS[query_transformer_type](
            **kwargs
        )
    else:
        raise ValueError(f"Unknown query transformer type: {query_transformer_type}")