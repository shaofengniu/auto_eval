from enum import Enum
from pydantic import BaseModel
from abc import ABC, abstractmethod
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.hyde.prompts import PROMPT_MAP
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
from termcolor import colored


class QueryTransformerType(str, Enum):
    DEFAULT_QUERY_TRANSFORMER = "default_transformer"
    HYDE_QUERY_TRANSFORMER = "hyde_query_transformer"
    KEYWORDS_QUERY_TRANSFORMER = "keywords_transformer"


class QueryTransformer(BaseModel, ABC):
    llm: BaseLanguageModel
    @abstractmethod
    def transform(self, query: str) -> str:
        pass


class DefaultQueryTransformer(QueryTransformer):
    def transform(self, query: str) -> str:
        return query
    

class HydeQueryTransformer(QueryTransformer):
    def transform(self, query: str) -> str:
        raise NotImplementedError("HydeQueryTransformer not implemented")


class KeywordsQueryTransformer(QueryTransformer):
    prompt_template: str = "Optimize the following query to make the most effective use of Google:\n{query}\nGoogle Search Keywords:"

    def transform(self, query: str) -> str:
        prompt = PromptTemplate(
            input_variables=["query"],
            template=self.prompt_template
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.predict(query=query)
        return result.replace('"', "")


QUERY_TRANSFORMER_TYPE_TO_CLASS = {
    QueryTransformerType.DEFAULT_QUERY_TRANSFORMER: DefaultQueryTransformer,
    QueryTransformerType.HYDE_QUERY_TRANSFORMER: HydeQueryTransformer,
    QueryTransformerType.KEYWORDS_QUERY_TRANSFORMER: KeywordsQueryTransformer,
}


def initialize_query_transformer(
    llm: BaseLanguageModel, query_transformer_type: QueryTransformerType, **kwargs
) -> QueryTransformer:
    if query_transformer_type in QUERY_TRANSFORMER_TYPE_TO_CLASS:
        return QUERY_TRANSFORMER_TYPE_TO_CLASS[query_transformer_type](
            llm=llm, **kwargs
        )
    else:
        raise ValueError(f"Unknown query transformer type: {query_transformer_type}")