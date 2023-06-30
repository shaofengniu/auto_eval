from chromadb import Client
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from nomic import atlas
import numpy as np

settings = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory="./index/chroma_index"
)

client = Client(settings)
print(client.list_collections())
embeddings_func = OpenAIEmbeddings().embed_documents
collection = client.get_collection(name="langchain", embedding_function=embeddings_func)

documents = collection.get(include=["embeddings", "metadatas"])
print(documents.keys())


embeddings = np.array(documents["embeddings"])
metadatas = documents["metadatas"]
ids = documents["ids"]
data = []
for i, id in enumerate(ids):
    data.append({"id": id.split("-")[0],
                 "source": metadatas[i]["source"],
                 "title": metadatas[i]["title"]})

with open("embeddings.tsv", "w") as f:
    for vector in embeddings:
        line = "\t".join([f"{v}" for v in vector]) + "\n"
        f.write(line)

with open("metadatas.tsv", "w") as f:
    f.write("id\tsource\ttitle\n")
    for metadata in data:
        f.write(f"{metadata['id']}\t{metadata['source']}\t{metadata['title']}\n")


exit()

project = atlas.map_embeddings(embeddings=embeddings,
                               data=data,
                               id_field="id")
print(project.maps)