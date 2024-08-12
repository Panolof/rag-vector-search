# src/milvus_setup.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Connect to Milvus using URI
uri = "http://127.0.0.1:19530"
connections.connect(uri=uri)

# Define the schema
fields = [
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="category", dtype=DataType.STRING)
]
schema = CollectionSchema(fields, "News Category Embeddings")

# Create a collection
collection = Collection("news_category", schema=schema)

# Load embeddings and categories
import numpy as np
embeddings = np.load('data/embeddings.npy')
categories = np.loadtxt('data/metadata.csv', delimiter=',', dtype=str, usecols=[0], skiprows=1)

# Insert data into the collection
collection.insert([embeddings, categories])

# Create an index on the vector field
collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}})

# Load the collection to memory
collection.load()
