# Data importer for AIDA and Handwritten Datasets


# Get batch of images:
import cv2
import os
import numpy as np
from enum import Enum

class SYMBOLS(Enum):
    COMMA = 0
    EXCLAMATION = 1
    PLUS = 2
    MINUS = 3
    LPAREN = 4
    RPAREN = 5
    ZERO = 6
    
def get_symbol_from_string(s):
    if s == ',':
        return SYMBOLS.COMMA
    elif s == '!':
        return SYMBOLS.EXCLAMATION
    elif s == '+':
        return SYMBOLS.PLUS
    elif s == '-':
        return SYMBOLS.MINUS
    elif s == '(':
        return SYMBOLS.LPAREN
    elif s == ')':
        return SYMBOLS.RPAREN
    elif s == '0':
        return SYMBOLS.ZERO
    else:
        raise ValueError('Invalid symbol string')
    
def open_image(path):
    """Open image from path

    Args:
        path (string): path to image

    Returns:
        Mat: image
    """    
    img = cv2.imread(path)
    return img


def get_aida_batch(batch_id):
    """AIDA batch getter, returns grayscale images as a generator

    Args:
        batch_id (int): batch id, the batch id is the number in the folder name

    Yields:
        Mat: image
    """        
    path = os.path.join('aida dataset',f'batch_{batch_id}','background_images')
    img_names = os.listdir(path)
    for img_name in img_names:
        img_path = os.path.join(path,img_name)
        img = open_image(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        yield img_gray
        
def get_handwritten_batch(batch_size):
    """Gets a batch of handwritten images and their symbols

    Args:
        batch_size (int): how many images to get

    Yields:
        tuple[SYMBOLS,Mat]: The Label and the image
    """    
    path = os.path.join('handwritten_math','extracted_images')
    img_types = os.listdir(path)
    for _ in range(batch_size):
        img_type = np.random.choice(img_types)
        img_type_path = os.path.join(path,img_type)
        img_names = os.listdir(img_type_path)
        img_name = np.random.choice(img_names)
        img_path = os.path.join(img_type_path,img_name)
        img = open_image(img_path)
        symbol = get_symbol_from_string(img_type)
        yield (symbol,img)

def show_img(img):
    """Displays an image

    Args:
        img (Mat): image to display
    """    
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()