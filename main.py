import cv2
from tracker import *


tracker =  EuclideanDistTracker() #This will track the detected objects in the camera

cap = cv2.VideoCapture("highway.mp4")
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) # threshold will exclude some of the frames that is assumed as moving
# this will extract only the moving objects from the camera

while True:
    ret, frame = cap.read()
    #lets extract region of interest in the video'
    height, width, _ = frame.shape
    print(height,width)
    roi = frame[240:720, 500:900]
    mask = object_detector.apply(roi)
    _,mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # this finds the cooridates of the white or moving objects in the camera

    detections = list()
    for cnt in contours:
        area = cv2.contourArea(cnt) # we calculate the area of the contour
        if area > 100: # we use if statement in order to remove small objects
            #cv2.drawContours(roi,[cnt],-1,(0,255,0),2) #here we use green color in order to draw countours of the moving objects
            x,y,w,h = cv2.boundingRect(cnt)
            detections.append([x,y,w,h])


    #Object Tracking
    box_ids = tracker.update(detections)
    for box_id in box_ids:
        x,y,w,h,id = box_id
        cv2.putText(roi,str(id),(x,y-10),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
    #cv2.imshow("Roi",roi)
    cv2.imshow("Frame", frame)
    #cv2.imshow("Mask",mask) #this makes everything that moving as white, the rest as black
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.release()
cv2.destroyWindow()