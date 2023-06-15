
from typing import List
from pydantic import BaseModel
from llama_index.retrievers import BaseRetriever as LlamaBaseRetriever

from llama_index.indices.base import BaseIndex
from langchain.schema import Document, BaseRetriever

DEFAULT_PERSIST_PATH = "llama_index"


class LlamaRetriever(BaseRetriever, BaseModel):
    index: BaseIndex
    retriever: LlamaBaseRetriever
    
    class Config:
        arbitrary_types_allowed = True

    def persist(self, path: str = DEFAULT_PERSIST_PATH) -> None:
        self.index.storage_context.persist(path)

    def get_relevant_documents(self, query: str) -> List[Document]:
        nodes = self.retriever.retrieve(query)
        docs = []
        for n in nodes:
            d = Document(page_content=n.node.get_text(),
                         metadata=n.node.node_info)
            d.metadata['score'] = n.score
            docs.append(d)
        docs.sort(key=lambda doc: doc.metadata['score'], reverse=True)
        return docs

    async def aget_relevant_documents(self, docs: List[Document]) -> List[Document]:
        raise NotImplementedError
    
    
