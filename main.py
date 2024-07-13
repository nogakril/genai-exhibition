import time
from concurrent.futures import ThreadPoolExecutor
import requests
import concurrent.futures
from openai import OpenAI
import pandas as pd

API_KEY = "7nPb-Hg5usk-AxmDfrc4kAQ7uXU5r1OIvwL9ozJ1D3VEQTkmYY-PBg"
OPENAI_API_KEY = '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'
MET_API_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{objectID}"
NUM_RESULTS = 5
openai_client = OpenAI(api_key=OPENAI_API_KEY)


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


# Get a random sample of object IDs with images
def show_sample_data(file_path, output_file_path, sample_size=100):
    object_ids_df = pd.read_csv(file_path)
    random_object_ids = object_ids_df.sample(n=sample_size)
    random_object_ids.to_csv(output_file_path, index=False)


# count how many records have images and are highlights
def count_highlights_with_images():
    data = pd.read_csv('MetObjects_w_hasImage.csv')
    count = data[(data['Is Highlight'] == True) | (data['hasImage'] == True)].shape[0]
    print(f"Number of records that are highlights and have images: {count}")


if __name__ == '__main__':
    test_object_ids = [39, 35, 180, 177, 166]

    curatorial_text, images_list = get_curatorial_text_and_images(test_object_ids)
    download_all_images(images_list)
    print(curatorial_text)

    # data = pd.read_csv('objects_w_images_thin_w_description.csv')
    # head = data.head(10)
    # print(data.head(10))
