import os
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import requests
import concurrent.futures
from openai import OpenAI
import pandas as pd
from bs4 import BeautifulSoup

API_KEY = "7nPb-Hg5usk-AxmDfrc4kAQ7uXU5r1OIvwL9ozJ1D3VEQTkmYY-PBg"
OPENAI_API_KEY = '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'
ACCESS_CLUSTER = "curl \
    -X GET 'https://6263858f-a18e-418f-b989-b0d27e6b20fb.us-east4-0.gcp.cloud.qdrant.io:6333' \
    --header 'api-key: 7nPb-Hg5usk-AxmDfrc4kAQ7uXU5r1OIvwL9ozJ1D3VEQTkmYY-PBg'"
MET_API_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{objectID}"
NUM_RESULTS = 5
db_client = QdrantClient(url="http://localhost:6333")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
COLUMNS_TO_VECTORIZE = ["Department", "Object Name", "Title", "Culture", "Period", "Artist Display Name",
                        "Artist Display Bio", "Object Date", "Medium", "Dimensions", "Classification", "Tags",
                        "Is Highlight"]


def create_collection():
    db_client.create_collection(
        collection_name="test_collection",
        vectors_config=VectorParams(size=4, distance=Distance.DOT),
    )


def fetch_metadata(id):
    response = requests.get(MET_API_URL.format(objectID=id))
    return response.json()


# Given a list of ids, make API calls to the metAPI to get the metadata for each id, return a list of metadata objects
def get_metadata(ids):
    metadata = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_metadata, id) for id in ids]
        for future in concurrent.futures.as_completed(futures):
            metadata.append(future.result())

    return metadata


# Extract the image URL, artist name, and title from the metadata
def extract_metadata(metadata):
    id = metadata['objectID']
    image_url = metadata['primaryImage']
    artist_name = metadata['artistDisplayName']
    title = metadata['title']
    return image_url, artist_name, title, id


# Create a message for ChatGPT to generate exhibition introductory text for an exhibition containing the artworks
def generate_message_and_context(metadata):
    messages = []
    images = {}
    text = ("Generate a very short exhibition introductory text for an exhibition "
            "containing the following artworks: {works}. "
            "Think of the concept, theme, and narrative that connects these artworks. "
            "Do not dive into deatils about the artworks.")
    for data in metadata:
        image_url, artist_name, title, id = extract_metadata(data)
        if len(image_url) > 0:
            messages.append(f"{title} by the artist {artist_name}")
            images[id] = image_url
    return text.format(works=", ".join(messages)), "You are a museum curator", images


# Create a query for OpenAI to generate exhibition introductory text for an exhibition containing the artworks
def generate_completion_request(message, model_context):
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": model_context},
                {"role": "user", "content": message},
            ],
            temperature=0,
            max_tokens=500,
        )
    except Exception as e:  # This catches any other exceptions
        print("Caught an exception:", e)
        completion = ""
    return completion.choices[0].message.content


def get_curatorial_text_and_images(ids):
    metadata = get_metadata(ids)
    message, context, images = generate_message_and_context(metadata)
    return generate_completion_request(message, context), images


def download_image(id, url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        save_path = f"images/{id}.jpg"
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open the file in binary write mode and write the content from the response
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Image downloaded successfully and saved at: {save_path}")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_all_images(images):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_image, id, url) for id, url in images.items()]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def get_object_ids_with_images():
    base_url = "https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&q="
    letters = "abcdefghijklmnopqrstuvwxyz"
    all_object_ids = set()

    for letter in letters:
        url = f"{base_url}{letter}"
        response = requests.get(url)
        if response.status_code == 200:
            object_ids = response.json().get('objectIDs', [])
            all_object_ids.update(object_ids)
        # Adding a small delay to avoid hitting the API rate limit
        time.sleep(0.1)
    # Save in a file
    output_file_path = 'object_ids_with_images.csv'
    pd.DataFrame(list(all_object_ids), columns=['objectID']).to_csv(output_file_path, index=False)
    return all_object_ids


def add_has_image_column():
    object_ids_with_images = get_object_ids_with_images()
    file_path = '/Users/noga.kril/Projects/bezalel/openaccess/MetObjects.csv'
    data = pd.read_csv(file_path)
    data['hasImage'] = data['Object ID'].apply(lambda x: x in object_ids_with_images)
    output_file_path = 'MetObjects_w_hasImage.csv'
    data.to_csv(output_file_path, index=False)


# Get a random sample of object IDs with images
def show_sample_data():
    object_ids_df = pd.read_csv('MetObjects_w_hasImage.csv')
    random_object_ids = object_ids_df.sample(n=100)
    random_object_ids.to_csv('temp.csv', index=False)


# count how many records have images and are highlights
def count_highlights_with_images():
    data = pd.read_csv('MetObjects_w_hasImage.csv')
    count = data[(data['Is Highlight'] == True) | (data['hasImage'] == True)].shape[0]
    print(f"Number of records that are highlights and have images: {count}")


# Function to fetch and parse HTML description using a session
def fetch_description(session, url, class_name):
    try:
        # Fetch the HTML content from the URL
        response = session.get(url)
        if response.status_code != 200:
            return None

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the div with the specified class name
        element = soup.find('div', class_=class_name)
        if element is None:
            return None

        # Extract and return the text content of the description
        description = element.get_text(strip=True)
        return description
    except Exception as e:
        return None


# Function to process a chunk of data using a session
def process_chunk(chunk, class_name, session):
    chunk['Description'] = chunk['Link Resource'].apply(
        lambda x: fetch_description(session, x, class_name) if pd.notna(x) else None)
    return chunk[['objectID', 'Link Resource', 'Description']]


# Function to save progress to a temporary CSV file
def save_progress(temp_file, chunk, mode='a'):
    if not os.path.exists(temp_file) or mode == 'w':
        chunk.to_csv(temp_file, index=False, mode=mode)
    else:
        chunk.to_csv(temp_file, index=False, mode=mode, header=False)


# Function to add description column using threading
def add_description_column(class_name='artwork__intro__desc', temp_file='temp_progress.csv',
                           output_file='MetObjects_w_hasImage_w_description.csv', chunksize=50, max_workers=10):
    data = pd.read_csv('MetObjects_w_hasImage.csv')

    # Filter rows where hasImage is True and keep only relevant columns
    data_with_images = data[data['hasImage'] == True][['Object ID', 'Link Resource']]

    # Remove temp file if it exists
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # Process data in chunks using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        with requests.Session() as session:
            for start in range(0, len(data_with_images), chunksize):
                end = start + chunksize
                chunk = data_with_images.iloc[start:end].copy()
                futures.append(executor.submit(process_chunk, chunk, class_name, session))

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                chunk_result = future.result()
                save_progress(temp_file, chunk_result)
                print(f"Progress: {((i + 1) * chunksize) / len(data_with_images) * 100:.2f}%")

    # Load the processed data from the temp file
    final_data = pd.read_csv(temp_file)

    # Merge the description data back into the original DataFrame
    data.set_index('Object ID', inplace=True)
    final_data.set_index('Object ID', inplace=True)
    data.update(final_data)
    data.reset_index(inplace=True)

    # Save the final output file
    data.to_csv(output_file, index=False)
    print(f"Updated CSV saved to '{output_file}'")


if __name__ == '__main__':
    # test_object_ids = [
    #     436524,
    #     484935,
    #     437112,
    #     210191,
    #     431264,
    # ]
    # curatorial_text, images_list = get_curatorial_text_and_images(test_object_ids)
    # download_all_images(images_list)
    add_description_column(chunksize=50, max_workers=20)
