#----------------------------Libraries---------------------------------#
import numpy as np
from numpy import array as ar
import cv2
from matplotlib import pyplot as plt

#------------------------Function Definition---------------------------#
first_iteration=True
finger_ct_history=[0,0]

# Function 0

def bg_subtract(image):
    fgmask = bg.apply(image)
    kernel = np.ones((3,3),np.uint8)
    fgmask = cv2.erode(fgmask,kernel,iterations = 1)
    frame=cv2.bitwise_and(image,image,mask=fgmask)
    return image
    
# Function 1

def histogram_calc(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #selecting the pixels in the rectangles
    square1 = hsv[125:150,275:300]
    square2 = hsv[175:200,275:300]
    square3 = hsv[225:250,275:300]
    square4 = hsv[125:150,325:350]
    square5 = hsv[175:200,325:350]
    square6 = hsv[225:250,325:350]

    #Creating ROI subset
    sq=np.concatenate((square1,square2,square3,square4,square5,square6))
    
    

    # Histogram Calculation
    histo = cv2.calcHist([sq],[0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(histo, histo, 0, 255, cv2.NORM_MINMAX)
    return histo

#Function 2

def back_proj(image,histo):
    image=cv2.medianBlur(image,3)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    back_projection = cv2.calcBackProject([hsv],[0,1],histo,[00,180,0,256],1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(back_projection,-1,disc,back_projection)
    back_projection=cv2.GaussianBlur(back_projection,(11,11),0)
    back_projection=cv2.medianBlur(back_projection,3)
    ret, thresh = cv2.threshold(back_projection, 50,255,0)
    kernel = np.ones((5,5), np.uint8)
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imshow('Dilated Image', dilate)
    return dilate

# Function 3

def hand_contour_find(contours):
    max_area=0
    largest_contour=-1
    for i in range(len(contours)):
        cnt=contours[i]
        area=cv2.arcLength(cnt,0)
        if(area>max_area):
            max_area=area
            largest_contour=i
    if(largest_contour==-1):
        return False,0
    else:
        h_contour=contours[largest_contour]
        return True,h_contour

# Function 4

def mark_hand_center(image,cnt):
    max_distance = 0
    pt = (0,0)
    x,y,w,h = cv2.boundingRect(cnt)
    for ind_y in xrange(int(y+0.3*h),int(y+0.8*h)): 
        for ind_x in xrange(int(x+0.3*w),int(x+0.6*w)):
            dist= cv2.pointPolygonTest(cnt,(ind_x,ind_y),True)
            if(dist>max_distance):
                max_distance=dist
                pt=(ind_x,ind_y)
    if(max_distance>0.04*image.shape[1]):
        thresh_score=True
        cv2.circle(image,pt,int(max_distance),(255,0,0),2)
    else:
        thresh_score=False
    return image,pt,max_distance,thresh_score

# Function 5
def mark_fingers(frame_in,hull,pt,radius):
    global first_iteration
    global finger_ct_history
    finger=[(hull[0][0][0],hull[0][0][1])]
    j=0

    cx = pt[0]
    cy = pt[1]
    
    for i in range(len(hull)):
        dist = np.sqrt((hull[-i][0][0] - hull[-i+1][0][0])**2 + (hull[-i][0][1] - hull[-i+1][0][1])**2)
        if (dist>18):
            if(j==0):
                finger=[(hull[-i][0][0],hull[-i][0][1])]
            else:
                finger.append((hull[-i][0][0],hull[-i][0][1]))
            j=j+1
    
    temp_len=len(finger)
    i=0
    while(i<temp_len):
        dist = np.sqrt( (finger[i][0]- cx)**2 + (finger[i][1] - cy)**2)
        if(dist<2.0*radius or dist>3.8*radius or finger[i][1]>cy+radius):
            finger.remove((finger[i][0],finger[i][1]))
            temp_len=temp_len-1
        else:
            i=i+1        
    
    temp_len=len(finger)
    if(temp_len>5):
        for i in range(1,temp_len+1-5):
            finger.remove((finger[temp_len-i][0],finger[temp_len-i][1]))
    
    palm=[(cx,cy),radius]

    if(first_iteration):
        finger_ct_history[0]=finger_ct_history[1]=len(finger)
        first_iteration=False
    else:
        finger_ct_history[0]=0.34*(finger_ct_history[0]+finger_ct_history[1]+len(finger))

    if((finger_ct_history[0]-int(finger_ct_history[0]))>0.8):
        finger_count=int(finger_ct_history[0])+1
    else:
        finger_count=int(finger_ct_history[0])

    finger_ct_history[1]=len(finger)

    count_text="FINGERS:"+str(finger_count)
    cv2.putText(frame_in,count_text,(int(0.62*frame_in.shape[1]),int(0.88*frame_in.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)

    for k in range(len(finger)):
        cv2.circle(frame_in,finger[k],10,255,2)
        cv2.line(frame_in,finger[k],(cx,cy),255,2)
    return frame_in,finger,palm

def image_capture(hist_captured):
    import cv2
    import numpy as np
    import math
    import time


    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        # read image
        ret, img = cap.read()

        # get hand data from the rectangle sub window on the screen
        cv2.rectangle(img, (300,300), (100,100), (0,255,0),0)
        crop_img = img[100:300, 100:300]

        # convert to grayscale
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # applying gaussian blur
        value = (35, 35)
        blurred = cv2.GaussianBlur(grey, value, 0)

        # thresholdin: Otsu's Binarization method
        _, thresh1 = cv2.threshold(blurred, 127, 255,
                                   cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # show thresholded image
        cv2.imshow('Thresholded', thresh1)

        # check OpenCV version to avoid unpacking error
        (version, _, _) = cv2.__version__.split('.')

        if version == '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
                   cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif version == '2':
            contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
                   cv2.CHAIN_APPROX_NONE)

        # find contour with max area
        cnt = max(contours, key = lambda x: cv2.contourArea(x))

        # create bounding rectangle around the contour (can skip below two lines)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt)

        # drawing contours
        drawing = np.zeros(crop_img.shape,np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt, returnPoints=False)

        # finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

        # applying Cosine Rule to find angle for all defects (between fingers)
        # with angle > 90 degrees and ignore defects
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]

            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

            # ignore angles > 90 and highlight rest with red dots
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0,0,255], -1)
            #dist = cv2.pointPolygonTest(cnt,far,True)

            # draw a line from start to end i.e. the convex points (finger tips)
            # (can skip this part)
            cv2.line(crop_img,start, end, [0,255,0], 2)
            #cv2.circle(crop_img,far,5,[0,0,255],-1)

        #function to click image
        def capture():
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('frame',frame)
                cv2.waitKey()
                #return_value,image = camera.read()
                #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                #horizontal_img = cv2.flip( image, 1)
                #cv2.imshow('image',horizontal_img)
                #if cv2.waitKey(1)& 0xFF == ord('s'):
                 #   cv2.imwrite('test.jpg',image)
                  #  break


        # define actions required
        if count_defects == 1:
            cv2.putText(img,"I am Vipul", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 2:
                    for y in range(0, 5):
                        time.sleep(1)
                        print('a')
                    ret, pic = cap.read()
                    cv2.imshow('pic',pic)
                    cv2.imwrite('hello.jpg',pic)

      

        # show appropriate images in windows
        cv2.imshow('Gesture', img)
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
#---------------------------Program------------------------------------#
cap = cv2.VideoCapture(0)

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

bg_cap = 0
histogram_cap = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame=cv2.bilateralFilter(frame,5,50,100)
    frame1 = np.copy(frame)
    frame_original = np.copy(frame)

    # Rectangles on left column
    cv2.rectangle(frame,(275,125),(300,150),(0,255,0),1)
    cv2.rectangle(frame,(275,175),(300,200),(0,255,0),1)
    cv2.rectangle(frame,(275,225),(300,250),(0,255,0),1)
    # Rectangles on right column
    cv2.rectangle(frame,(325,125),(350,150),(0,255,0),1)
    cv2.rectangle(frame,(325,175),(350,200),(0,255,0),1)
    cv2.rectangle(frame,(325,225),(350,250),(0,255,0),1)
    
    if(bg_cap):
        bgsubtract = bg_subtract(frame)

    if (not (bg_cap and histogram_cap)):
        if(not(bg_cap)):
            cv2.putText(frame,"Remove hand from the frame and press 'B' to capture background",(int(0.05*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,8)
        else:
            cv2.putText(frame,"Place hand inside boxes and press 'H' to capture hand histogram",(int(0.08*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,8)
            first_iteration = True
            finger_ct_history = [0,0]
    else:
        backproj = back_proj(bgsubtract,hist_captured)
        frame_copy = np.copy(frame)
        _,contours,hierarchy = cv2.findContours(backproj,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        found,hand_contour = hand_contour_find(contours)
        if(found):
            cv2.drawContours(frame,contours,-1,(128,255,0),3)
            hand_convex_hull=cv2.convexHull(hand_contour)
            frame,hand_center,hand_radius,hand_size_score=mark_hand_center(frame_original,hand_contour)
            if(hand_size_score):
                frame,finger,palm=mark_fingers(frame,hand_convex_hull,hand_center,hand_radius)
            else:
                frame = frame_original
            
    cv2.imshow('Hand Gesture Recognition v1.0',frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        image_capture(frame)
    #Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #Press C to capture histogram
    elif cv2.waitKey(1) & 0xFF == ord('h'):
        hist_captured = histogram_calc(frame)
        histogram_cap =1
    #Press B to capture background
    elif cv2.waitKey(1) & 0xFF == ord('b'):
        bg = cv2.createBackgroundSubtractorMOG2(0,10)
        bg_cap = 1
    #Press R to Reset
    elif cv2.waitKey(1) & 0xFF == ord('r'):
        histogram_cap = 0
        bg_cap = 0
            
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
