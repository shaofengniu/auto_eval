from enum import Enum
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

class ModelType(str, Enum):
    CHAT_OPENAI = "chat_openai"


class EmbeddingType(str, Enum):
    OPENAI_EMBEDDINGS = "openai_embeddings"



MODEL_TYPE_TO_CLASS = {
    ModelType.CHAT_OPENAI: ChatOpenAI
}


def initialize_model(model_type: ModelType, **kwargs):
    if model_type in MODEL_TYPE_TO_CLASS:
        return MODEL_TYPE_TO_CLASS[model_type](**kwargs)
    raise ValueError(f"Unknown model type: {model_type}")


EMBEDDING_TYPE_TO_CLASS = {
    EmbeddingType.OPENAI_EMBEDDINGS: OpenAIEmbeddings
}

def initialize_embedding(embedding_type: EmbeddingType, **kwargs):
    if embedding_type in EMBEDDING_TYPE_TO_CLASS:
        return EMBEDDING_TYPE_TO_CLASS[embedding_type](**kwargs)
    raise ValueError(f"Unknown embedding type: {embedding_type}")
