import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import pywt

cA, cD = pywt.dwt([1, 2, 3, 4], 'db1')

print(cA, cD)

help(pywt)