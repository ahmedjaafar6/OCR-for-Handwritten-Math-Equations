# Main file for project
import cv2
from matplotlib import pyplot as plt
import numpy as np
import utils
import statistics

def find_good_contours_thres(img, conts, alpha = 0.005):
    '''+  
    Function to find threshold of good contours on basis of 10% of maximum area
    Input: Contours, threshold for removing noises
    Output: Contour area threshold
    
    For image dim img.shape
    '''
    #Calculating areas of contours and appending them to a list
    areas = []
    
    for c in conts:
        areas.append([cv2.contourArea(c)**2])
        
    #alpha is controlling parameter 
    areas = np.asarray([x[0] for x in areas])
    print("Percentile : ", str(np.percentile(areas, 75)))
    thres = alpha * np.percentile(areas, 75)
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

    (thresh, erosion) = cv2.threshold(erosion, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find the contours
    if(cv2.__version__ == '3.3.1'):
        xyz, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #Find a contour threshold for this dataset (hyperparam!)
    contour_threshold = find_good_contours_thres(erosion, contours, alpha=alpha)

    print("thresh: ", str(contour_threshold))

    contours_thresh = []
    for c in contours:       
        if( cv2.contourArea(c)**2 > contour_threshold):
            
            contours_thresh.append(c)

    #Retrieved bounding boxes
    contours_sorted, bounding_boxes = sort_contours(contours_thresh,method="left-to-right")

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

    images = Parser(img, 0.005)

    for im in images:
        plt.imshow(im, cmap="gray")
        plt.show()

