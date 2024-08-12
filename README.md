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

## Setting Up Milvus

To use this project, you need to have Milvus installed and running. The easiest way to set up Milvus is by using Docker. Follow these steps:

### Prerequisites

- **Docker**: Ensure you have Docker installed on your system. You can download and install Docker from [here](https://www.docker.com/products/docker-desktop).

### Steps to Set Up Milvus

1. **Download the Official Milvus Docker Compose File**:
   In your terminal, run the following command to download the Milvus standalone Docker Compose file:

   ```bash
   wget https://github.com/milvus-io/milvus/releases/download/v2.0.2/milvus-standalone-docker-compose.yml -O docker-compose.yml
   ```

   This command will download the `docker-compose.yml` file directly from the Milvus GitHub repository for version 2.0.2.

2. **Start Milvus**:
   In the terminal, navigate to the directory where you downloaded the `docker-compose.yml` file and run:

   ```bash
   docker-compose up -d
   ```

   This command will start the Milvus service in detached mode.

3. **Verify Milvus is Running**:
   You can check if Milvus is running with:

   ```bash
   docker-compose ps
   ```
    After Milvus standalone starts, there will be three Docker containers running, including the Milvus standalone service and its two dependencies.

4. **Stopping Milvus**:
   When you are done, you can stop the Milvus service with:

   ```bash
   docker-compose down
   ```
   If you want to delete the data after stopping Milvus, run:
   ```bash
   sudo rm -rf volumes
   ```
For more detailed instructions, refer to the official Milvus documentation: [Milvus Standalone Installation](https://milvus.io/docs/v2.0.x/install_standalone-docker.md).


### Connecting to Milvus

The application will automatically connect to Milvus using the environment variables specified in your `.env` file. Ensure you have the correct values set for `MILVUS_HOST` and `MILVUS_PORT`.

Example `.env` configuration:

```env
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
OPENAI_API_KEY=your_openai_api_key
```

By following these steps, you will have a fully operational Milvus instance running in Docker, ready to be used with this project.


### Setting Up Environment Variables
To securely manage API keys and database connections, this project uses a .env file to store environment variables. Follow the steps below to set it up:

1. Create a .env File
* In the root directory of your project, create a new file named .env.
* You can use the .env.example file as a reference. Run the following command to copy it:
```bash
cp .env.example .env
```

2. Fill in the .env File
Open the newly created .env file and replace the placeholder values with your actual credentials:
* OPENAI_API_KEY: This is your OpenAI API key. You can obtain it from [OpenAI's API](https://beta.openai.com/signup/).
```plaintext
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
* MILVUS_HOST: This is the host address for your Milvus vector database. If you are running Milvus locally, this will likely be 127.0.0.1.
```plaintext
MILVUS_HOST=127.0.0.1
```
* MILVUS_PORT: This is the port number for your Milvus vector database. The default port is 19530.
```plaintext
MILVUS_PORT=19530
```
3. Save the .env File
After filling in the values, save the .env file. This file will now be used to configure the API key and database connections throughout the project.
Note: Never commit your .env file to version control. It contains sensitive information that should remain private. This should be documented in the [.gitignore](.gitignore)




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
