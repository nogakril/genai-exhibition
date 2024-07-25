import os
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd

from objects_generation import fetch_metadata


# Function to fetch and parse HTML description
def fetch_description(url, class_name):
    try:
        # Fetch the HTML content from the URL
        response = requests.get(url)
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


def fetch_metadata_in_parallel(ids, max_workers=10):
    has_image = [False] * len(ids)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(fetch_metadata, object_id): index for index, object_id in
                           enumerate(ids) if pd.notna(object_id)}

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                if result and 'primaryImage' in result and result['primaryImage']:
                    has_image[index] = True
            except Exception as e:
                print(f"Error processing metadata for index {index}: {e}")

    return has_image


def fetch_descriptions_in_parallel(links, class_name, max_workers=10):
    descriptions = [None] * len(links)
    processed_flags = [False] * len(links)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(fetch_description, link, class_name): index for index, link in
                           enumerate(links) if pd.notna(link)}

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                descriptions[index] = future.result()
                processed_flags[index] = True
            except Exception as e:
                print(f"Error fetching description for index {index}: {e}")

    return descriptions, processed_flags


# Function to add a new column "Description" to the CSV file
def add_description_column(input_file, output_file):
    chunksize = 50

    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):
        links = chunk['Link Resource'].tolist()
        descriptions, processed_flags = fetch_descriptions_in_parallel(links, 'artwork__intro__desc')
        chunk['Description'] = descriptions
        chunk['Processed'] = processed_flags

        mode = 'w' if i == 0 else 'a'
        header = (i == 0)

        chunk.to_csv(output_file, mode=mode, header=header, index=False)

        print(f"Processed and saved chunk {i + 1}")

    print(f"Updated CSV saved to '{output_file}'")


def add_image_presence_column(input_file, output_file):
    chunksize = 1000
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):
        object_ids = chunk['Object ID'].tolist()
        has_image = fetch_metadata_in_parallel(object_ids)

        chunk['Has Image'] = has_image

        mode = 'w' if i == 0 else 'a'
        header = (i == 0)

        chunk.to_csv(output_file, mode=mode, header=header, index=False)

        print(f"Processed and saved chunk {i + 1}")

    print(f"Updated CSV saved to '{output_file}'")


# def add_has_image_column(file_path, output_file_path):
#     object_ids_with_images = get_object_ids_with_images()
#     data = pd.read_csv(file_path)
#     data['hasImage'] = data['Object ID'].apply(lambda x: x in object_ids_with_images)
#     data.to_csv(output_file_path, index=False)


def vlookup(input_file, lookup_file, output_file):
    data = pd.read_csv(input_file)
    lookup_data = pd.read_csv(lookup_file)
    data['Has Image'] = data['Object ID'].map(lookup_data.set_index('Object ID')['Has Image'])
    data = data[data['Has Image'] == True]
    data.to_csv(output_file, index=False)


if __name__ == '__main__':
    vlookup('/Users/noga.kril/PycharmProjects/genAiExhibition/MetObjects_w_hasImageTrue_w_description.csv',
            '/Users/noga.kril/PycharmProjects/genAiExhibition/objects_w_images_thin_v2.csv',
            '/Users/noga.kril/PycharmProjects/genAiExhibition/MetObjects_w_hasImageTrue_w_description_w_description.csv')
