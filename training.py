import sys
import numpy as np
import cv2

# Load extracted image of paring codes, infinite numbers in one block only
im = cv2.imread('code_image.jpg')
im3 = im.copy()

# Preprocess image
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

# Find contours
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

samples =  np.empty((0,400))
responses = []
keys = [i for i in range(48,58)]

for cnt in contours:
    if cv2.contourArea(cnt)>100:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if h>45 and h<55:
            im4 = im.copy()
            cv2.rectangle(im4,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(20,20))
            cv2.imshow('norm',im4)
            key = cv2.waitKey()
            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,400))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"

# Save training results to file
np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
