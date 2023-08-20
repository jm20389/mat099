import os
import sys
import shutil
from PIL import Image

# Import classes package from parent directory
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(script_directory)
sys.path.append(parent_directory)

from classes import *

output_dataset_dir = './datasets/data-korus/serialized/'

DatasetProcessor.remove_duplicate_files(output_dataset_dir + '/tampered/')
DatasetProcessor.remove_duplicate_files(output_dataset_dir + '/natural/')

print('completed')