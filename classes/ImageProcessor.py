import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from matplotlib import pyplot as plt

class ImageProcessor:

    @staticmethod
    def convertToELA(path, quality=90):
        temp_filename = 'temp_file_name.jpg'
        ela_filename = 'temp_ela.png'

        image = Image.open(path).convert('RGB')
        image.save(temp_filename, 'JPEG', quality=quality)

        temp_image = Image.open(temp_filename).convert(image.mode)  # Ensure the same color space as the original image
        ela_image = ImageChops.difference(image, temp_image)

        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        return ela_image

    @staticmethod
    def convert_to_ela_image(path,quality):
        original_image = Image.open(path).convert('RGB')

        #resaving input image at the desired quality
        resaved_file_name = 'resaved_image.jpg'     #predefined filename for resaved image
        original_image.save(resaved_file_name,'JPEG',quality=quality)
        resaved_image = Image.open(resaved_file_name)

        #pixel difference between original and resaved image
        ela_image = ImageChops.difference(original_image,resaved_image)

        #scaling factors are calculated from pixel extremas
        extrema = ela_image.getextrema()
        max_difference = max([pix[1] for pix in extrema])
        if max_difference ==0:
            max_difference = 1
        scale = 350.0 / max_difference

        #enhancing elaimage to brighten the pixels
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        ela_image.save("ela_image.png")
        return ela_image

