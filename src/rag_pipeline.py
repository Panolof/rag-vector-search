import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pymilvus import connections, Collection
import pandas as pd

# Load environment variables from the .env file
load_dotenv()

# Access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
milvus_host = os.getenv("MILVUS_HOST")
milvus_port = os.getenv("MILVUS_PORT")

# Connect to Milvus
connections.connect("default", host=milvus_host, port=milvus_port)

# Load the collection
collection_name = "news_category"
collection = Collection(collection_name)
collection.load()

# Print collection info for debugging
print(f"Collection schema: {collection.schema}")
print(f"Number of entities: {collection.num_entities}")

# Check if index exists
index_list = collection.indexes
if index_list:
    print("Existing indexes:")
    for index in index_list:
        print(f"Field: {index.field_name}, Index params: {index.params}")
else:
    print("No indexes found on the collection.")

# Create a vector store with a 384-dimensional embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Milvus(
    collection_name=collection_name,
    embedding_function=embeddings,
    connection_args={"host": milvus_host, "port": milvus_port},
    text_field="category",
    vector_field="embedding"
)

# Create a custom prompt template
prompt_template = """You are an AI assistant specialized in analyzing news trends. Use the following pieces of context to answer the question at the end. If the context doesn't provide relevant information to answer the question, say so and provide a general response based on your knowledge.

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Set up RAG with a language model
llm = OpenAI(temperature=0.2, openai_api_key=openai_api_key)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 10}),  # Retrieve 10 documents
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Function to process query and print results
def process_query(query):
    print(f"\nQuery: {query}")
    response = qa({"query": query})
    print("Generated Response:", response['result'])
    print("\nRetrieved Documents:")
    for doc in response['source_documents']:
        print(f"- Category: {doc.page_content}")
        if hasattr(doc, 'metadata'):
            print(f"  Metadata: {doc.metadata}")
    print("\n" + "="*50 + "\n")

# Example queries
queries = [
    "What are the most discussed topics in U.S. NEWS?",
    "Can you summarize recent trends in TECH news?",
    "What are the main themes in BUSINESS news articles?",
    "How has political coverage changed over time based on the POLITICS category?",
    "What are the popular topics in the WELLNESS category?"
]

for query in queries:
    process_query(query)