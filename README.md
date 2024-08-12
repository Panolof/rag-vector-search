# RAG Vector Search with Milvus using News Category Dataset

This repository demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline using a Milvus vector database and a pre-trained language model. The project is based on the "News Category Dataset" and showcases how to preprocess the data, generate embeddings, and perform vector search.

## Project Structure
```plaintext
rag-vector-search/
├── src/
│   ├── preprocess.py        # Script to preprocess data and create embeddings
│   ├── milvus_setup.py      # Script to set up Milvus and insert vectors
│   ├── rag_pipeline.py      # Script to create the RAG pipeline
│   ├── app.py               # Flask app for UI interaction
├── data/                    # Directory to store embeddings and metadata
│   ├── embeddings.npy       # File to store the generated embeddings
│   ├── metadata.csv         # File to store metadata (category, headline, short_description)
├── templates/
│   ├── index.html           # HTML template for Flask app
├── docs/                    # Documentation directory
│   ├── innerworkings.md     # Detailed explanation of the repository's inner workings
├── README.md                # Project overview and step-by-step guide
├── .gitignore               # Files and directories to ignore in the repository
└── requirements.txt         # Python dependencies
```

## Getting Started

### Prerequisites
Ensure you have the following installed on your system:
* Python 3.8 or higher
* pip (Python package installer)
* Virtual environment setup (optional but recommended)

### Installation
1. Clone the Repository: 
```bash
git clone https://github.com/your-username/rag-vector-search.git
cd rag-vector-search

```

2. Create a Virtual Environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
.\venv\Scripts\activate   # On Windows=
```

3. Install Dependencies:
```bash
pip install -r requirements.txt
```
## Step-by-Step Workflow
1. Preprocessing the Dataset:
The src/preprocess.py script loads the "News Category Dataset", preprocesses the text data, and generates embeddings using a pre-trained language model. These embeddings are saved as a NumPy array for later use.
* Dataset Download: The script downloads the dataset from the Hugging Face hub.
* Text Embedding: Each headline and description is combined and transformed into a vector using a pre-trained model.
* Saving Data: The embeddings are saved to data/embeddings.npy, and the corresponding metadata is saved to data/metadata.csv.

To run the preprocessing script:
```bash
python src/preprocess.py
```

Expected Output:
* The embeddings and metadata will be stored in the data/ directory.
* Progress will be shown in the terminal, including the time taken for each step.

2. Setting Up Milvus:
The src/milvus_setup.py script connects to a Milvus instance, creates a collection for the embeddings, and inserts the vectors into the database.
* Milvus Connection: Ensure Milvus is running locally or on a remote server.
* Vector Insertion: The embeddings are indexed and stored in Milvus for efficient vector search.

To set up Milvus and insert the vectors:
```bash
python src/milvus_setup.py
```

3. Building the RAG Pipeline:
The src/rag_pipeline.py script ties everything together by creating a RAG pipeline that can perform queries against the vector database and generate responses using a language model.
* Querying: Enter a query to search the vector database and get generated responses.
* Language Model Integration: The script uses the language model to provide contextually relevant answers based on vector search results.

To run the RAG pipeline:
```bash
python src/rag_pipeline.py
```

4. Interacting with the Flask App:

The src/app.py script provides a simple web interface to interact with the RAG pipeline. The app allows users to enter queries and view generated responses.

To start the Flask app: 
```bash
python src/app.py
```

Visit http://127.0.0.1:5000/ in your browser to use the app.

## Documentation
A detailed explanation of the inner workings of this repository is provided in the [docs/innerworkings.md](docs/innerworkings.md) file. This includes information on:
* Data Processing: How the dataset is preprocessed and transformed into embeddings.
* Vector Search: How Milvus is used to store and retrieve vectors efficiently.
* RAG Pipeline: Integration of vector search with a language model to create a RAG pipeline.
* UI Interaction: How the Flask app is structured to interact with the pipeline.

## License
This project is licensed under the MIT License.

## Contributions
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions for improvements.
