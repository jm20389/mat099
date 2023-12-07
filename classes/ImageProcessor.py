import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
from matplotlib import pyplot as plt
import requests, re
from io import BytesIO
import tensorflow as tf

from classes.StyleTransferProcessor   import StyleTransferProcessor

class ImageProcessor(StyleTransferProcessor):

    @staticmethod
    def manipulateImage(im, manipulation):
        if manipulation is None or not isinstance(manipulation, dict):
            return im

        operation = manipulation['operation']
        parameter = manipulation['parameter']

        # Basic Image manipulations
        if operation == 'brightness':
            enhancer = ImageEnhance.Brightness(im)
            im = enhancer.enhance(parameter)
        elif operation == 'contrast':
            enhancer = ImageEnhance.Contrast(im)
            im = enhancer.enhance(parameter)
        elif operation == 'saturation':
            # Perform saturation changes (convert to HSV, modify, and convert back to RGB)
            im = im.convert("HSV")
            im = ImageEnhance.Color(im).enhance(parameter)
            im = im.convert("RGB")

        # Generative - Style Transfer
        elif operation == 'style_transfer':
            myStylePath = "/resources/styles/Pierre-Auguste_Renoir_42.jpg"
            im = ImageProcessor.styleTransferImage(im, myStylePath, style_weight=1e-2, content_weight=1e4)

        # Image filters
        elif operation == 'clarendon':
            im = ImageProcessor.apply_clarendon(im, parameter)
        elif operation == 'juno':
            im = ImageProcessor.apply_juno(im, parameter)
        elif operation == 'gingham':
            im = ImageProcessor.apply_gingham(im, parameter)
        elif operation == 'lark':
            im = ImageProcessor.apply_lark(im, parameter)
        elif operation == 'sierra':
            im = ImageProcessor.apply_sierra(im, parameter)
        elif operation == 'ludwig':
            im = ImageProcessor.apply_ludwig(im, parameter)

        return im

    # Filter implementations
    @staticmethod
    def apply_clarendon(im, intensity):
        # Clarendon filter: Increase contrast and saturation
        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(1.3 * intensity)
        enhancer = ImageEnhance.Color(im)
        im = enhancer.enhance(1.2 * intensity)
        return im

    @staticmethod
    def apply_juno(im, intensity):
        # Juno filter: Apply a bluish tint
        im = ImageEnhance.Color(im).enhance(1.5)
        return ImageOps.colorize(im.convert("L"), "#4878b6", "#e381b4").convert(im.mode)

    @staticmethod
    def apply_gingham(im, intensity):
        # Gingham filter: Apply a warm tone
        enhancer = ImageEnhance.Color(im)
        im = enhancer.enhance(1.3 * intensity)
        return ImageEnhance.Brightness(im).enhance(1.1 * intensity)

    @staticmethod
    def apply_lark(im, intensity):
        # Lark filter: Increase brightness and apply a warm tone
        enhancer = ImageEnhance.Brightness(im)
        im = enhancer.enhance(1.2 * intensity)
        im = ImageEnhance.Color(im).enhance(1.1 * intensity)
        return im
    @staticmethod
    def apply_sierra(im, intensity):
        # Sierra filter: Apply a vintage look with sepia tones
        grayscale_im = im.convert('L')
        sepia_im = Image.merge('RGB', [
            ImageEnhance.Brightness(grayscale_im).enhance(0.9 * intensity),
            ImageEnhance.Contrast(grayscale_im).enhance(1.1 * intensity),
            ImageEnhance.Color(grayscale_im).enhance(1.3 * intensity)
        ])
        sepia_im = ImageEnhance.Contrast(sepia_im).enhance(1.2)
        sepia_im = ImageEnhance.Brightness(sepia_im).enhance(0.9)
        return sepia_im
    @staticmethod
    def apply_ludwig(im, intensity):
        # Ludwig filter: Apply a cool tone
        im = ImageEnhance.Color(im).enhance(0.8 * intensity)
        return ImageEnhance.Brightness(im).enhance(1.2 * intensity)

    # Style Transfer implementation
    @staticmethod
    def styleTransferImage(im, style_path,style_weight, content_weight):
        processorOB = StyleTransferProcessor(#myContentPath
                                     #,myStylePath
                                     saveDir =           None
                                     ,max_dim =          512
                                     ,resizeContentImg = False
                                     ,cropContentImg =   True)

        transformedImage = processorOB.sequence(
                    #  content_path = None
                    # ,style_image =  None
                     content_path =   im
                    ,style_image =    style_path
                    ,content_layers = ['block5_conv2']
                    ,style_layers =   ['block1_conv1',
                                      'block2_conv1',
                                      'block3_conv1',
                                      'block4_conv1',
                                      'block5_conv1']
                    ,style_weight =   style_weight
                    ,content_weight = content_weight
                    ,train_steps =    1
                    ,opt =            tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
                    ,longer_opt =     True
                    )

        return transformedImage

    @staticmethod
    def openImage(directory, local = True):
        if local:
            return Image.open(directory).convert('RGB')
        try:
            response = requests.get(directory)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                return img
            else:
                print(f"Failed to retrieve the image from {directory}. Status code: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

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

    def resize_images_in_folder(self, input_folder, output_folder, width):

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        files = os.listdir(input_folder)

        supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif', '.bmp', '.webp', '.ico', '.jfif', '.jp2', '.jpx', '.j2k', '.j2c', '.pgm', '.pbm', '.pnm', '.ras', '.spp', '.tga', '.xbm', '.blp', '.cur', '.dds', '.dib', '.emf', '.fits', '.hdr', '.svg', '.rgb', '.sgi', '.icns')

        for file in files:
            input_path = os.path.join(input_folder, file)
            if file.lower().endswith(supported_formats):
                img = Image.open(input_path)

                # Calculate the new height to maintain the aspect ratio
                width_percent = (width / float(img.size[0]))
                height = int((float(img.size[1]) * float(width_percent)))

                img = img.resize((width, height), Image.ANTIALIAS)
                output_path = os.path.join(output_folder, file)
                img.save(output_path)


