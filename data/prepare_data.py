import os
import pandas as pd

def generate_flickr8k_data(data_folder):
    image_folder = os.path.join(data_folder, 'Flicker8k_Dataset')
    captions_file = os.path.join(data_folder, 'Flickr8k.token.txt')

    # Read captions from the file
    with open(captions_file, 'r') as file:
        captions_data = file.readlines()

    image_captions = []

    # Process each line in the captions file
    for line in captions_data:
        parts = line.strip().split('\t')
        image_id, caption = parts[0].split("#")[0], parts[1]
        image_path = os.path.join(image_folder, image_id)

        # Append image path and caption to the list
        image_captions.append({'image_path': image_path, 'caption': caption})

    # Create a DataFrame
    df = pd.DataFrame(image_captions)

    # Save the DataFrame to a CSV file
    df.to_csv('flickr8k_data.csv', index=False)

if __name__ == "__main__":
    data_folder = '/path/to/your/flickr8k/dataset'
    generate_flickr8k_data(data_folder)
