import torch
from transformers import BertModel, BertTokenizer
from qdrant_client import QdrantClient
from qdrant_client.http.models import models, PointStruct
import pandas as pd

# Constants
COLUMNS_TO_VECTORIZE = ["Object Name", "Title", "Tags", "Description", "Department", "Medium", "Artist Display Name",
                        "Object Begin Date", "Classification", "Is Highlight"]
BATCH_SIZE = 1000
VECTOR_DIM = 768  # BERT base produces vectors of size 768
TOP_K = 9
db_client = QdrantClient(url="http://localhost:6333")
collection_name = "museum_objects"

# Load pre-trained BERT model and tokenizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device).eval()


def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Is Highlight'] = df['Is Highlight'].replace({True: 'highlight', False: ''})
    df.fillna('', inplace=True)
    return df


def create_collection(db_client, collection_name):
    config = models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE)
    # collection_exists = db_client.collection_exists(collection_name=collection_name)
    # if collection_exists:
    #     db_client.delete_collection(collection_name=collection_name)
    db_client.create_collection(collection_name=collection_name, vectors_config=config)
    print(f"Collection '{collection_name}' created with vector size {VECTOR_DIM}.")


def vectorize_text(text):
    """Convert text to a vector using BERT."""
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded_input)
    vectors = outputs.last_hidden_state.detach().cpu().numpy().max(axis=1).tolist()[0]  # Verify axis, check max
    return vectors


def vectorize_and_save_batch(batch, db_client, batch_index, total_batches, collection_name):
    print(f"Processing batch {batch_index + 1} of {total_batches}...")

    # Prepare points for insertion
    points = []
    for index, row in batch.iterrows():
        # Create a single string from the relevant columns for vectorization
        text = ' '.join('[' + str(col) + '] ' + str(row[col]).lower()  for col in COLUMNS_TO_VECTORIZE)
        vector = vectorize_text(text)

        # Create PointStruct, ensuring 'Object ID' is retrieved directly as row['Object ID']
        point = PointStruct(vector=vector, id=row['Object ID'])
        points.append(point)

    # Insert points into Qdrant
    db_client.upsert(points=points, collection_name=collection_name)
    print(f"Completed batch {batch_index + 1} of {total_batches}.")


def process_data(df, db_client):
    batches = [df.iloc[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    total_batches = len(batches)
    for batch_index, batch in enumerate(batches):
        vectorize_and_save_batch(batch, db_client, batch_index, total_batches, collection_name)


def vectorization_process(filepath):
    create_collection(db_client, collection_name)
    df = load_data(filepath)
    print(f"Starting processing of {len(df)} rows...")
    process_data(df, db_client)
    print("All data has been processed and saved.")


def vectorize_input_text(text):
    """Convert a single input text to a vector using BERT."""
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}  # Move to GPU if available
    with torch.no_grad():
        outputs = model(**encoded_input)
    return outputs.last_hidden_state.cpu().numpy().max(axis=1).tolist()[0]


def search_similar_vectors(vector, top_k):
    """Search for the top similar vectors in the database."""
    results = db_client.search(collection_name=collection_name, search_params=models.SearchParams(hnsw_ef=128, exact=True),
                               query_vector=vector, limit=top_k)
    return [result.id for result in results]


def get_top_objects(input_text, top_k=TOP_K):
    vector = vectorize_input_text(input_text)  # Vectorize input text
    similar_object_ids = search_similar_vectors(vector, top_k)  # Search for similar vectors
    print(f"Top {top_k} similar object IDs: {similar_object_ids}")
    return similar_object_ids


if __name__ == "__main__":
    vectorization_process('MetObjects_w_hasImageTrue_w_description.csv')  # Replace with your actual file path
