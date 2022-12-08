# Main file for project
import cv2
from matplotlib import pyplot as plt

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

def Parser(img, alpha): 

    L_H = img.shape[0]

    ## apply some dilation and erosion to join the gaps - turn thick contours into lines
    #Selecting elliptical element for dilation    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilation = cv2.dilate(img,kernel,iterations = 2)
    erosion = cv2.erode(dilation,kernel,iterations = 1)

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
    
    char_locs = []

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 
    i = 0
    char_type =[]
    while i in range(0, len(contours_sorted)):
    
        x,y,w,h = bounding_boxes[i]
        exp = 0
        if i+1 != len(contours_sorted):
            x1,y1,w1,h1 = bounding_boxes[i+1]
            if abs(x-x1) < 10 and  (h1+h) < 70:
                #print(h+h1)
                minX = min(x,x1)
                minY = min(y,y1)
                maxX = max(x+w, x1+w1)
                maxY = max(y+h, y1+h1)
                x,y,x11,y11 = minX, minY, maxX, maxY
                
                x,y,w,h = x,y,x11-x,y11-y
                i = i+2
                continue
        
        #char_locs.append([x,y,x+w,y+h])     
        if(h<0.10*L_H and w<0.10*L_H):
            #print('Yes')
            i=i+1
            continue

        char_locs.append([x-2,y+Y1-2,x+w+1,y+h+Y1+1,w*h]) #Normalised location of char w.r.t box image
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(153,180,255),2)
        if i!=0:
            if y+h < (L_H*(1/2)) and y < bounding_boxes[i-1][1] and h < bounding_boxes[i-1][3]:
                exp = 1
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        i = i+1
        char_type.append(exp)
    
    if(show == True):        
        plt.figure(figsize=(15,8))    
        plt.axis("on")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    df_char = pd.DataFrame(char_locs)
    df_char.columns=['X1','Y1','X2','Y2','area']
    df_char['exp'] = char_type
    df_char['pred'] = df_char.apply(lambda c: predict(dict_clean[box_num],c['X1'],\
           c['Y1'],c['X2'], c['Y2'], acc_thresh=acc_thresh), axis=1 )
    df_char['pred_proba'] = df_char.apply(lambda c: predict(dict_clean[box_num],c['X1'],\
           c['Y1'],c['X2'], c['Y2'], proba=True, acc_thresh=acc_thresh), axis=1 )
    df_char['line_name'] = line_name
    df_char['box_num'] = box_num
    return [box_num,line_name,df_char]
    pass



if __name__ == "__main__":
    exit()