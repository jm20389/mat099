# Worker to rename files
import os
from PIL import Image

def convert_png_to_jpg(input_path, output_path):
    try:
        with Image.open(input_path) as img:
            img.convert("RGB").save(output_path, "JPEG")
        print(f"Conversion successful. JPEG image saved at {output_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")



nat_dir = "/mnt/669118d5-25c6-4d9a-9660-2787d5d59e99/vision_dataset/nat/"
output_path = "/mnt/669118d5-25c6-4d9a-9660-2787d5d59e99/vision_dataset/stye_transfer_device_pictures_51-56/"
os.makedirs(output_path, exist_ok=True)

# print(os.listdir(nat_dir))

for picture in os.listdir(nat_dir):

    if picture.split('.')[-1] == 'png':
        convert_png_to_jpg(nat_dir + picture, output_path + picture.split('.')[0] + '.jpg')