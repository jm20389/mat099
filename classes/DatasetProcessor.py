import numpy as np
#from PIL import Image, ImageChops, ImageEnhance
from PIL import Image

import os, sys
from glob import glob
from multiprocessing import cpu_count, Pool
import pandas as pd
import urllib.request
from matplotlib import pyplot as plt
import hashlib
import subprocess
import re, requests
import seaborn as sns

# EDA
import cv2
import re
from scipy.stats import skew, kurtosis


class DatasetProcessor():

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

    @staticmethod
    def remote_vision_dataset():
        url = 'https://lesc.dinfo.unifi.it/PrnuModernDevices/dataset_download.txt'

        try:
            response = requests.get(url)
            if response.status_code == 200:
                links = response.text.splitlines()

                def findDevice(url):
                    pattern = r'https:\/\/lesc[^\n]+\/[^\n]+\/([A-Z]{1,2}[0-9]{1,3})'
                    return re.search(pattern, url).group(1)

                def flatOrNat(url):
                    pattern = r'https:\/\/lesc[^\n]+\/[^\n]+\/[A-Z]{1,2}[0-9]{1,3}\/(flat|nat)'
                    try:
                        capture = re.search(pattern, url).group(1)
                        return capture
                    except:
                        return None

                ff_dirlist = np.array(sorted([link for link in links if flatOrNat(link) == 'flat']))
                ff_device = np.array(set([findDevice(link) for link in links]))

                nat_dirlist = np.array(sorted([link for link in links if flatOrNat(link) == 'nat']))
                nat_device = np.array(set([findDevice(link) for link in links if flatOrNat(link) == 'nat']))

                return ff_dirlist, ff_device, nat_dirlist, nat_device

            else:
                print(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    @staticmethod
    def verify_vision_dataset_directory(directory):
        directory = directory.split('/')[-1]
        pattern = r'^([A-Z]{1,2}[0-9]{1,3}_I_[0-9]+)'
        if re.search(pattern, directory):
            return True
        else:
            return False

    @staticmethod
    def local_vision_dataset(ff_dir, nat_dir, deviceList):
        clean_list = []
        for device in deviceList:
            match = re.search(r'^(D[0-9]{1,2})', device)
            if match:
                device = match.group(1) + '_I'
                clean_list.append(device)
        deviceList = clean_list

        for directory in [ff_dir, nat_dir]:
            if directory[-1] != '/' and ('.' in directory.split('/')[-1]):
                extension = directory.split('/')[-1]
                directory.replace(extension, '/')

            directory += '*.[Jj][Pp][EeGgTt][GgIi][FfNn]'

        ff_dirlist =  sorted(glob(ff_dir))
        ff_device =   [os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist]
        nat_dirlist = sorted(glob(nat_dir))
        nat_device =  [os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist]

        # Verify filename structure:
        nat_dirlist = [filepath for filepath in nat_dirlist if DatasetProcessor.verify_vision_dataset_directory(filepath)]
        if nat_dirlist == []:
            print('Invalid filename format for natural pictures directory.')
            quit()

        def findDevice(directory):
            filename = directory.split('/')[-1]
            pattern = r'^([A-Z]{1,2}[0-9]{1,3}_I)'
            match_result = re.search(pattern, filename)
            if match_result:
                return match_result.group(1)
            else:
                pattern = r'^([A-Z]{1,2}[0-9]{1,3})'
                return re.search(pattern, filename).group(1)

        if (deviceList is not None and isinstance(deviceList, list)):
            ff_dirlist =  [directory for directory in ff_dirlist  if findDevice(directory) in deviceList]
            ff_device =   [directory for directory in ff_device   if findDevice(directory) in deviceList]
            nat_dirlist = [directory for directory in nat_dirlist if findDevice(directory) in deviceList]
            nat_device =  [directory for directory in nat_device  if findDevice(directory) in deviceList]

            if nat_dirlist == []:
                print('Natural Images directory does not contain any image from the provided device list.\n Provided device list: ')
                print(deviceList)
                quit()
        else:
            # Intersect:
            ff_dirlist =  [directory for directory in ff_dirlist  if findDevice(directory) in nat_device]
            ff_device =   [directory for directory in ff_device   if directory in set(nat_device)]

        ff_dirlist =  np.array(ff_dirlist)
        ff_device =   np.array(ff_device)
        nat_dirlist = np.array(nat_dirlist)
        nat_device =  np.array(nat_device)

        return ff_dirlist, ff_device, nat_dirlist, nat_device








