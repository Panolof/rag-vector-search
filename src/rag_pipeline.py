import os
from dotenv import load_dotenv
from langchain import MilvusVectorStore, RetrievalAugmentedGeneration
from langchain.llms import OpenAI
from pymilvus import connections, Collection

# Load environment variables from the .env file
load_dotenv()

# Access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
milvus_host = os.getenv("MILVUS_HOST")
milvus_port = os.getenv("MILVUS_PORT")

# Connect to Milvus
connections.connect("default", host=milvus_host, port=milvus_port)
collection = Collection("news_category")

# Create a vector store
vector_store = MilvusVectorStore.from_collection(collection, "embedding")

# Set up RAG with a language model
rag = RetrievalAugmentedGeneration(
    llm=OpenAI(api_key=openai_api_key),
    vector_store=vector_store
)

# Example query
query = "Latest trends in AI for business"
response = rag.query(query)
print("Generated Response:", response)
