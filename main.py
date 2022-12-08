# Main file for project
import cv2
from matplotlib import pyplot as plt
import numpy as np
import utils

def find_good_contours_thres(conts, alpha = 0.002):
    '''+  
    Function to find threshold of good contours on basis of 10% of maximum area
    Input: Contours, threshold for removing noises
    Output: Contour area threshold
    
    For image dim 3307*4676
    alpha(text_segment) = 0.01
    alpha(extract_line) = 0.002
    '''
    #Calculating areas of contours and appending them to a list
    areas = []
    
    for c in conts:
        areas.append([cv2.contourArea(c)**2])
    #alpha is controlling paramter    
    thres = alpha * max(areas)[0]
    
    return thres

def sort_contours(contours, method="left-to-right"):
    '''
    sort_contours : Function to sort contours
    argument:
        contours (array): image contours
        method(string) : sorting direction
    output:
        contours(list): sorted contours
        boundingBoxes(list): bounding boxes
    '''
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (contours, boundingBoxes)

def Parser(img, alpha, show=True): 

    ## apply some dilation and erosion to join the gaps - turn thick contours into lines
    #Selecting elliptical element for dilation    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilation = cv2.dilate(img,kernel,iterations = 2)
    erosion = cv2.erode(dilation,kernel,iterations = 1)

    # Convert to grayscale
    erosion = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    plt.imshow(erosion)
    plt.show()
    
    # Find the contours
    if(cv2.__version__ == '3.3.1'):
        xyz, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
    #Find a contour threshold for this dataset (hyperparam!)
    contour_threshold = find_good_contours_thres(contours, alpha)
    contours = []
    for c in contours:       
        if( cv2.contourArea(c)**2 > contour_threshold):
            contours.append(c)
    
    #Retrieved bounding boxes
    contours_sorted, bounding_boxes = sort_contours(contours,method="left-to-right")

    
    if(show == True):        
        plt.figure(figsize=(15,8))    
        plt.axis("on")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    extracted_symbols = []
    for i, box in enumerate(bounding_boxes):
        x, y, l, h = box
        extracted_symbols.append(img[y : y + h, x : x + l])

    return extracted_symbols

if __name__ == "__main__":
    img = utils.get_aida_batch(1)
    for i in img:
        plt.imshow(i)
        plt.show()
        img = i
        break

    images = Parser(img, 0.05)

    for im in images:
        plt.imshow(im, cmap="gray")
        plt.show()

