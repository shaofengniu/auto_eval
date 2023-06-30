from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="index/chroma_index", embedding_function=embeddings
)
retriever = vectorstore.as_retriever()

while True:
    query = input("Query: ")
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print("-" * 80)
        print("* " + doc.metadata["source"])
        print("* " + doc.metadata["title"])
        print(doc.page_content)