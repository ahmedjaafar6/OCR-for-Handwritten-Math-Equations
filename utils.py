# Data importer for AIDA and Handwritten Datasets


# Get batch of images:
import cv2
import os
import numpy as np

_handwritten_key = dict()
_handwritten_path = os.path.join('datasets','handwritten')
_aida_path = os.path.join('datasets','aida')
_mnist_path = os.path.join('datasets','mnist')
_handwritten_paths = []

def get_handwritten_keys():
    """Gets mappings between symbols and their integer representation

    Returns:
        tuple[dict[str,int],list[str]]: A dict that maps strings to integers and a list of strings at the same index as the integer
    """    
    if not _handwritten_key:
        img_types = os.listdir(_handwritten_path)
        for i,img_type in enumerate(img_types):
            _handwritten_key[img_type] = i
    return _handwritten_key, img_types

    
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
    path = os.path.join(_aida_path,f'batch_{batch_id}','background_images')
    img_names = os.listdir(path)
    for img_name in img_names:
        img_path = os.path.join(path,img_name)
        img = open_image(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        yield img_gray
        
def get_handwritten_batch(batch_id,batch_count):
    """Gets a batch of handwritten images and their values, returns a generator.

    Args:
        batch_id (int): batch id (must be less than batch_count)
        batch_count (int): number of batches total

    Yields:
        tuple[str,Mat]: The Label and the image
    """    
    # Initialize paths
    if not _handwritten_paths:
        img_types = os.listdir(_handwritten_path)
        for img_type in img_types:
            img_type_path = os.path.join(_handwritten_path,img_type)
            img_names = os.listdir(img_type_path)
            for img_name in img_names:
                img_path = os.path.join(img_type_path,img_name)
                _handwritten_paths.append((img_type,img_path))
        np.random.shuffle(_handwritten_paths)
        
    # Get batch and return 
    batch = _handwritten_paths[batch_id*batch_count:(batch_id+1)*batch_count]
    for img_type,img_path in batch:
        img = open_image(img_path)
        yield (img_type,img)

def show_img(img):
    """Displays an image

    Args:
        img (Mat): image to display
    """    
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()