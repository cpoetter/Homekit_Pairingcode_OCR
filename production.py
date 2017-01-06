import cv2
import numpy as np
import collections
import sys

# Load training results
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

# Initialize machine learning with training results
model = cv2.KNearest()
model.train(samples,responses)

# Load image and preprocess it
im = cv2.imread('code_image.jpg')
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

# Find contours
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

pairing_code_numbers = {}
for cnt in contours:
    if cv2.contourArea(cnt)>100:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if h>45 and h<55:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(20,20))
            roismall = roismall.reshape((1,400))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
            string = str(int((results[0][0])))
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
            pairing_code_numbers[x] = string

# FindContours is sorting the detected numbers by size, sort them by position
if len(pairing_code_numbers) is not 8:
    raise NameError('No code detected.')

sorted_pairing_code_numbers = collections.OrderedDict(sorted(pairing_code_numbers.items()))
pairing_code_list = sorted_pairing_code_numbers.values()
pairing_code_list.insert(3, '-')
pairing_code_list.insert(6, '-')
pairing_code = ''.join(pairing_code_list)
print "The pairing code is: " + pairing_code

if len(sys.argv) > 1:
    debug_mode = sys.argv[1]
else:
    debug_mode = 'False'

# Show a window with found numbers and there value
if debug_mode == 'debug':
    print 'Debug mode is on, display result in window'
    cv2.imshow('im',im)
    cv2.imshow('out',out)
    cv2.waitKey(0)
