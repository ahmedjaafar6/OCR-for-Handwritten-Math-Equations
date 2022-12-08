# Data importer for AIDA and Handwritten Datasets


# Get batch of images:
import cv2

def open_image(path):
    img = cv2.imread(path)
    return img

open_image('aida dataset/batch1/background_images/1.jpg')