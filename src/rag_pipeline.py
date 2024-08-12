# src/rag_pipeline.py
from langchain import MilvusVectorStore, RetrievalAugmentedGeneration
from langchain.llms import OpenAI

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")
collection = Collection("news_category")

# Create a vector store
vector_store = MilvusVectorStore.from_collection(collection, "embedding")

# Set up RAG with a language model
rag = RetrievalAugmentedGeneration(
    llm=OpenAI(api_key="your_openai_api_key"),
    vector_store=vector_store
)

# Example query
query = "Latest trends in AI for business"
response = rag.query(query)
print("Generated Response:", response)
