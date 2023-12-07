from datetime                import datetime
from matplotlib              import pyplot as plt
from multiprocessing         import cpu_count, Pool
from glob                    import glob
from sklearn.model_selection import train_test_split
from skimage.io              import imread
from PIL                     import Image, ImageChops, ImageEnhance
from sklearn.metrics         import roc_curve, roc_auc_score

import os, sys, re, subprocess, hashlib, time, traceback, hashlib, configparser
import urllib.request
import numpy   as np
import pandas  as pd
import seaborn as sns
import shutil

import tensorflow as tf
import keras
from tensorflow.keras.layers     import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models     import Model
from tensorflow.keras.optimizers import SGD, Adam

from .PickleHandler          import PickleHandler
from .ImageProcessor         import ImageProcessor
from .ModelBuilder           import ModelBuilder
from .PRNUProcessor          import PRNUProcessor
from .PRNUManager            import PRNUManager
from .DatasetProcessor       import DatasetProcessor
from .Gan                    import GanBuilder
from .StyleTransferProcessor import StyleTransferProcessor
from .EDAHelper              import EDAHelper
from .WorkloadManager        import WorkloadManager
from .SQLiteManager          import SQLiteManager

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages

# Load config:
config_file_path = os.path.join(os.path.dirname(__file__), '../config.ini')
config = configparser.ConfigParser()
config.read(config_file_path)

username = config['credentials']['username']
password = config['credentials']['password']

# Use in the code
# print(f"Username: {username}")
# print(f"Password: {password}")





















