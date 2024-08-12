import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import time
from tqdm import tqdm

# Function to embed text using the model
def embed_text(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to print the elapsed time for a step
def print_time_taken(start_time, step_description):
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for {step_description}: {elapsed_time:.2f} seconds")

def main():
    # Start overall timing
    overall_start_time = time.time()

    # Step 1: Load the dataset
    start_time = time.time()
    ds = load_dataset("heegyu/news-category-dataset")
    print_time_taken(start_time, "loading dataset")

    # Convert dataset to DataFrame
    start_time = time.time()
    df = pd.DataFrame(ds['train'])
    print_time_taken(start_time, "converting dataset to DataFrame")

    # Step 2: Load the tokenizer and model
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    print_time_taken(start_time, "loading tokenizer and model")

    # Step 3: Create embeddings for headlines and descriptions
    start_time = time.time()
    df['embedding'] = df['headline'] + " " + df['short_description']

    # Apply embedding function with progress bar
    df['embedding'] = [embed_text(text, tokenizer, model) for text in tqdm(df['embedding'], desc="Creating embeddings")]
    print_time_taken(start_time, "creating embeddings")

    # Step 4: Save the preprocessed embeddings and metadata
    start_time = time.time()
    np.save('data/embeddings.npy', np.vstack(df['embedding'].values))
    df[['category', 'headline', 'short_description']].to_csv('data/metadata.csv', index=False)
    print_time_taken(start_time, "saving embeddings and metadata")

    # End overall timing
    print_time_taken(overall_start_time, "total processing time")

if __name__ == "__main__":
    main()
