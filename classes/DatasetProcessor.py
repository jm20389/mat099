import numpy as np
#from PIL import Image, ImageChops, ImageEnhance
from PIL import Image

import os
from glob import glob
from multiprocessing import cpu_count, Pool
import pandas as pd
import urllib.request
from matplotlib import pyplot as plt
import hashlib
import subprocess


class DatasetProcessor:

    def __init__(self, dataDir:str = None, max_width:int = 300):
        self.dataDir = './data-images/' if dataDir is None else dataDir
        self.max_width = max_width
        self.cameras = ['Sony_A57', 'Canon_60D', 'Nikon_D90', 'Nikon_D7000']

    def dirCollection(self):
        pics = []
        picDirs = []
        camera_model = []
        tampered = []

        for camera in self.cameras:
            for type in ['pristine', 'tampered-realistic']:
                dirContents =   os.listdir(self.dataDir + '/' + camera + '/' + type)
                picDirs +=      [self.dataDir + '/' + camera + '/' + type + '/' + picDir for picDir in dirContents]
                camera_model += [camera] * len(dirContents)
                tampered +=     [type] * len(dirContents)
                #pics +=         [np.array(Image.open(dataDir + '/' + camera + '/' + type + '/' + picDir)) for picDir in dirContents]
                #pics =          [np.array(Image.open(dir)) for dir in picDirs]
                for picDir in picDirs:
                    image = Image.open(picDir)
                    resized_pic = self.resizeSingle(image, self.max_width)
                    pics.append(resized_pic)

        return pics, picDirs, camera_model, tampered

    def resizeSingle(self, image, new_width:int) -> np.array :
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        max_size = (new_width, int(new_width / aspect_ratio))
        image = image.resize(max_size)

        return np.array(image)

    def resizePics(self, pics:list, new_width: int):
        # if ~all(isinstance(pic, np.ndarray) for pic in pics):
        #     print('All items must be numpy arrays')
        #     return None

        resizedPics = []
        for pic in pics:
            resized_pic = self.resizeSingle(Image.fromarray(pic), new_width)
            resizedPics.append(resized_pic)

        return resizedPics

    @staticmethod
    def get_file_hash(filename, block_size=65536):
        hasher = hashlib.md5()
        with open(filename, 'rb') as f:
            while True:
                data = f.read(block_size)
                if not data:
                    break
                hasher.update(data)
        return hasher.hexdigest()

    @staticmethod
    def remove_duplicate_files(directory):
        files = os.listdir(directory)
        files = sorted(files, reverse=True)

        hash_to_filenames = {}

        for filename in files:
            full_path = os.path.join(directory, filename)
            if os.path.isfile(full_path):
                file_hash = DatasetProcessor.get_file_hash(full_path)
                if file_hash in hash_to_filenames:
                    duplicate_filename = hash_to_filenames[file_hash]
                    duplicate_path = os.path.join(directory, duplicate_filename)
                    print(f"Removing duplicate: {full_path} (duplicate of {duplicate_path})")
                    try:
                        subprocess.run(['sudo', 'rm', full_path], check=True)
                        print(f"File '{full_path}' has been deleted.")
                    except subprocess.CalledProcessError as e:
                        print(f"Error deleting file '{full_path}': {e}")
                    #os.remove(full_path)
                else:
                    hash_to_filenames[file_hash] = filename








