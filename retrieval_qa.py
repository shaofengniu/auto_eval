from typing import Optional, Any
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema import BaseRetriever
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
from models import ModelType, initialize_model


def initialize_qa_chain(llm: BaseLanguageModel, prompt: Optional[PromptTemplate] = None, **kwargs: Any) -> LLMChain:
    _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
    llm_chain = LLMChain(llm=llm, prompt=_prompt)
    return llm_chain
