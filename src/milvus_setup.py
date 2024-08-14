from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np

# Connect to Milvus using URI
uri = "http://127.0.0.1:19530"
connections.connect(uri=uri)

print("Milvus server status:", utility.get_server_version())

# Define the schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=255)
]
schema = CollectionSchema(fields, "News Category Embeddings")

# Drop existing collection if it exists
if utility.has_collection("news_category"):
    utility.drop_collection("news_category")

# Create a collection
collection = Collection("news_category", schema=schema)

# Load embeddings and categories
embeddings = np.load('data/embeddings.npy')
categories = np.loadtxt('data/metadata.csv', delimiter=',', dtype=str, usecols=[0], skiprows=1)

print("Embeddings shape:", embeddings.shape)
print("Categories shape:", categories.shape)
print("Number of embeddings:", len(embeddings))
print("Number of categories:", len(categories))

# Ensure the number of embeddings and categories match
min_length = min(len(embeddings), len(categories))
embeddings = embeddings[:min_length]
categories = categories[:min_length]

# Truncate categories to 255 characters
categories = [cat[:255] for cat in categories]

# Insert data into the collection
insert_data = [
    embeddings.tolist(),
    categories
]

print("Length of embedding data:", len(insert_data[0]))
print("Length of category data:", len(insert_data[1]))

# Insert data in smaller batches
batch_size = 10000
for i in range(0, len(insert_data[0]), batch_size):
    batch_embeddings = insert_data[0][i:i+batch_size]
    batch_categories = insert_data[1][i:i+batch_size]
    try:
        collection.insert([batch_embeddings, batch_categories])
        print(f"Inserted batch {i//batch_size + 1}")
    except Exception as e:
        print(f"Error inserting batch {i//batch_size + 1}: {str(e)}")
        # You might want to log problematic entries here

# Create an index on the vector field
collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}})

# Load the collection to memory
collection.load()

# Print some statistics
print(f"Total entities in collection: {collection.num_entities}")

print("Milvus setup completed successfully!")