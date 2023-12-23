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
from classes import PickleHandler
import cv2
import re
from scipy.stats import skew, kurtosis


class EDAHelper(PickleHandler):

    @staticmethod
    def calculate_scalar_image_statistics(image_path, histogram = False):
        device =    re.search('D[0-9]{2}', image_path).group(0)
        picname =   re.search('D[0-9]{2}_I_([0-9]+)', image_path).group(1)
        format =    re.search('D[0-9]{2}_I_[0-9]+\.([A-Za-z]+)', image_path).group(1)
        img =       Image.open(image_path)
        img_array = np.array(img)

        stats = {
            'Device':        device
            ,'Picname':      picname
            ,'Format':       format
            ,'Size':         np.prod(img_array.shape[:2])
            ,'AvgIntensity': np.mean(img_array)
            ,'IntensityStd': np.std(img_array)
            ,'Brightness':   np.mean(img.convert('L'))
            ,'Contrast':     cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).var()
            ,'Sharpness':    cv2.Laplacian(img_array, cv2.CV_64F).var()
            ,'mean':         np.mean(img_array)
            ,'std_dev':      np.std(img_array)
            ,'min':          np.min(img_array)
            ,'max':          np.max(img_array)
            ,'median':       np.median(img_array)
            ,'skewness':     skew(img_array.flatten())
            ,'kurtosis':     kurtosis(img_array.flatten())
        }

        # Color Statistics
        for channel in range(img_array.shape[2]):
            stats[f'mean_channel_{channel}'] = np.mean(img_array[:, :, channel])
            stats[f'std_dev_channel_{channel}'] = np.std(img_array[:, :, channel])

        # Spatial Statistics
        stats['center_of_mass'] = np.array(np.unravel_index(np.argmax(img_array), img_array.shape)).mean()
        stats['image_width'] = img_array.shape[1]
        stats['image_height'] = img_array.shape[0]

        # Texture Statistics (using Sobel filter)
        sobel_x = np.abs(cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3))
        sobel_y = np.abs(cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3))
        stats['grad_std_dev'] = np.std(sobel_x + sobel_y)

        # Intensity Statistics
        if histogram:
            stats['intensity_histogram'] = np.histogram(img_array.flatten(), bins=256)[0]

        # Frequency Domain Statistics
        f_transform = np.fft.fft2(img_array)
        power_spectrum = np.abs(f_transform) ** 2
        stats['dominant_frequency'] = np.argmax(power_spectrum)

        stats['filename'] = image_path.split('/')[-1]

        return stats

    @staticmethod
    def feedPics(pickleObjectName, nat_pictures):

        df = PickleHandler.load(pickleObjectName)

        extracted_filenames =    list(df['filename'])
        remaining_nat_pictures = [picture for picture in nat_pictures if not picture.split('/')[-1] in extracted_filenames]

        remaining = len(remaining_nat_pictures)
        buffer = 10

        for filepath in remaining_nat_pictures:

            if filepath.split('/')[-1] in extracted_filenames:
                print('Picture ' + filepath + ' already extracted.')
                continue

            buffer = buffer - 1
            if buffer == 0:
                PickleHandler.save(df, 'data-pickle/' + pickleObjectName)
                buffer = 10

            try:
                statistics = DatasetProcessor.calculate_scalar_image_statistics(filepath)
                if remaining > 0:
                    remaining = remaining - 1

                index = len(df)
                df.loc[index] = statistics
                print('New row inserted: ' + str(len(df)) + '. For picture: ' + filepath.split('/')[-1])
                print('Remaining: ' + str(remaining))
            except:
                print('Error processing picture: ' + filepath)
                break

        PickleHandler.save(df, 'data-pickle/' + pickleObjectName)
        print('df saved as pickle object.')

        return None

    @staticmethod
    def plotPicsPerDevice(df):
        train_counts_balanced = df.Device.value_counts().sort_values(ascending = True)

        fig = plt.figure(figsize=(30, 25))
        ax=plt.subplot()

        plt.barh(range(len(train_counts_balanced)), train_counts_balanced.values, color = 'green')
        plt.title('Number of pictures per Device', fontsize=40)
        plt.xlabel('Number of pictures', fontsize = 20)
        plt.ylabel('Device', fontsize = 20)

        labels = list(df.Device.unique())

        ax.set_yticks(range(len(train_counts_balanced)))
        ax.set_yticklabels(labels)

        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        plt.show()

        return train_counts_balanced

    @staticmethod
    def intensityDistributionPerDevice(df, save=False):
        fig = plt.figure(figsize=(15, 20))

        devices = list(df.Device.unique())
        axes = [category + "_ax" for category in devices]

        for i in range(len(devices)):
            axes[i] = fig.add_subplot(9, 5, i + 1)

            current_device = devices[i]
            values = df[df.Device == current_device]["AvgIntensity"]

            sns.histplot(values, color="red")

            axes[i].set_xlim([0, 250])
            axes[i].set_title(devices[i], fontsize=15)

        fig.subplots_adjust(hspace=1)
        fig.suptitle("Distribution of picture average pixel intensity per device", fontsize=24, y=0.92)

        if save:
            plt.savefig("documentation/report/images/pixel_intensity_distribution.png")

        plt.show()

        return None

    @staticmethod
    def intensityBoxPlotPerDevice(df, save = False):
        # Calculate the pixel intensity mean for each picture in the train dataset:
        pic_means = list(df.AvgIntensity)

        df_pixels = pd.DataFrame( list(zip(pic_means, list(df.Device))), columns=['Picture avg', 'Device']  )

        median_order = df_pixels.groupby(by=["Device"])["Picture avg"].median().iloc[::-1].index

        fig = plt.figure(figsize=(25, 10))
        ax=plt.subplot()

        sns.boxplot(x = "Device", y = "Picture avg", palette = "YlOrBr", data = df_pixels, order = median_order)

        plt.title('Average pixel intensity per device', fontsize=30)
        plt.xlabel('Device')
        plt.ylabel('Average pixel intensity')
        plt.xticks(rotation = 90)

        if save:
            plt.savefig('' + "report/images/pixel_intensity_barplot.png")

        plt.show()

        return None








