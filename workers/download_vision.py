import requests, re
from PIL import Image
from io import BytesIO

# List of image URLs
#url = 'https://lesc.dinfo.unifi.it/PrnuModernDevices/dataset_download.txt'
url = 'https://lesc.dinfo.unifi.it/VISION/VISION_files.txt'
response = requests.get(url)
if response.status_code == 200:
    image_urls = response.text.splitlines()

# Directory where you want to save the images
#output_directory = "VISION_IMAGES/"
output_directory = "/mnt/669118d5-25c6-4d9a-9660-2787d5d59e99/vision_dataset/"

ff_directory  = "/mnt/669118d5-25c6-4d9a-9660-2787d5d59e99/vision_dataset/ff/"
nat_directory = "/mnt/669118d5-25c6-4d9a-9660-2787d5d59e99/vision_dataset/nat/"

# Create the output directory if it doesn't exist
import os
os.makedirs(output_directory, exist_ok=True)
os.makedirs(ff_directory, exist_ok=True)
os.makedirs(nat_directory, exist_ok=True)

# Download and save the images
for i, url in enumerate(image_urls):

    if (url.split('.')[-1]).lower() not in ['.jpg', 'jpeg', 'png'] :
        continue

    if "/flat/" in url:
        output_directory = ff_directory
    elif "/nat/" in url:
        output_directory = nat_directory
    else:
        continue

    try:

        filename = url.split('/')[-1].replace('_flat_', '_').replace('_nat_', '_')

        if filename in os.listdir(output_directory):
            print(f"File already exists: {filename}")

        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to download: {url}")
            continue

        image = Image.open(BytesIO(response.content))

        #image.save(os.path.join(output_directory, f"image_{i}.jpg"))
        image.save(os.path.join(output_directory, filename))
        print(f"Downloaded: {url}")

    except Exception as e:
        print(f"An error occurred while downloading {url}: {str(e)}")

print("Download complete.")
