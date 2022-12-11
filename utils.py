# Data importer for AIDA and Handwritten Datasets


# Get batch of images:
import matplotlib.pyplot as plt
import gzip
import cv2
import os
import numpy as np
# from sklearn.datasets import fetch_openml

_handwritten_key = dict()
_handwritten_path = os.path.join('datasets', 'handwritten')
_aida_path = os.path.join('datasets', 'aida')
_mnist_path = os.path.join('datasets', 'mnist')
_handwritten_paths = []


def get_key_length():
    get_handwritten_keys()
    return len(_handwritten_key)


def get_handwritten_keys(labels=[]):
    if not _handwritten_key:
        img_types = os.listdir(_handwritten_path)
        for i, img_type in enumerate(img_types):
            _handwritten_key[img_type] = i
    nums = []
    for label in labels:
        nums.append(_handwritten_key[label])
    return np.array(nums)


def open_image(path):
    """Open image from path

    Args:
        path (string): path to image

    Returns:
        Mat: image
    """
    img = cv2.imread(path)
    return img


def standardize_image(img, invert=False, resize=False, to_gray=False):
    """Standardize image by converting to grayscale and resizing

    Args:
        img (Mat): image
        invert (bool, optional): invert the image. Defaults to False.

    Returns:
        Mat: standardized image
    """
    SIZE = (28, 28)
    if resize:
        img = cv2.resize(img, SIZE)
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    if invert:
        img = 255 - img
    return img


def get_aida_batch(batch_id):
    """AIDA batch getter, returns grayscale images as a generator

    Args:
        batch_id (int): batch id, the batch id is the number in the folder name

    Yields:
        Mat: image
    """
    path = os.path.join(_aida_path, f'batch_{batch_id}', 'background_images')
    img_names = os.listdir(path)
    for img_name in img_names:
        img_path = os.path.join(path, img_name)
        img = open_image(img_path)
        img = standardize_image(img, to_gray=True)
        yield img


def get_handwritten_batch(batch_count, batch_min, batch_max=None):
    """Gets a batch of handwritten images and their values, returns a generator.

    Args:
        batch_count (int): number of batches total
        batch_min (int): batch id (must be less than batch_count)

    Yields:
        tuple[str,Mat]: The Label and the image
    """
    if batch_max is None:
        batch_max = batch_min + 1
    # Initialize paths
    if not _handwritten_paths:
        img_types = os.listdir(_handwritten_path)
        for img_type in img_types:
            img_type_path = os.path.join(_handwritten_path, img_type)
            img_names = os.listdir(img_type_path)
            for img_name in img_names:
                img_path = os.path.join(img_type_path, img_name)
                _handwritten_paths.append((img_type, img_path))
        np.random.shuffle(_handwritten_paths)

    batch_size = len(_handwritten_paths)//batch_count

    # Get batch and return
    batch = _handwritten_paths[batch_min * batch_size:batch_max*batch_size]
    # print(len(batch))
    labels = []
    images = []
    for img_type, img_path in batch:
        img = open_image(img_path)
        img = standardize_image(img, invert=True, to_gray=True, resize=True)
        labels.append(img_type)
        images.append(img)
    return np.array(labels), np.array(images)


def show_img(img):
    """Displays an image

    Args:
        img (Mat): image to display
    """
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_mnist():
    path = os.path.join(_mnist_path, 't10k-images-idx3-ubyte.gz')
    with gzip.open(path, 'r') as f:
        image_size = 28
        num_images = 10000
        f.read(16)
        buffer = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        print(data.shape)
        data = data.reshape(num_images, image_size, image_size)
    with gzip.open(os.path.join(_mnist_path, 't10k-labels-idx1-ubyte.gz'), 'r') as f:
        f.read(8)
        buffer = f.read(10000)
        labels = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
    return labels, data

# Evaluate the accuracy of labels


def evaluateLabels(y, ypred, visualize=True):

    classLabels = np.unique(y)
    conf = np.zeros((len(classLabels), len(classLabels)))
    for tc in range(len(classLabels)):
        for pc in range(len(classLabels)):
            conf[tc, pc] = np.sum(np.logical_and(y == classLabels[tc],
                                                 ypred == classLabels[pc]).astype(float))

    acc = np.sum(np.diag(conf))/y.shape[0]

    if visualize:
        plt.figure()
        plt.imshow(conf, cmap='gray')
        plt.ylabel('true labels')
        plt.xlabel('predicted labels')
        plt.title('Confusion matrix (Accuracy={:.2f})'.format(acc*100))
        plt.show()

    return (acc, conf)


def one_hot(y, num_classes):
    """One hot encoding of y

    Args:
        y (np.array): labels
        num_classes (int): number of classes

    Returns:
        np.array: one hot encoded labels
    """
    return np.eye(num_classes)[y]


if __name__ == '__main__':
    l, d = read_mnist()
    print(l.shape, d.shape)
    for _ in range(10):
        print(l[_])
        img = d[_]
        plt.imshow(img)
        plt.show()
