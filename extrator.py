import sys
import numpy as np
import cv2

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

# Get the image path
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    raise NameError('No image path given. Please provide the image path as a parameter.')

# Load a color image
orig = cv2.imread(image_path)
if orig is None:
    raise NameError('Image file does not exists.')

# iPhone images store rotating in header, for OpenCV the actual image need to be rotated
if orig.shape[0] < orig.shape[1]:
    print 'Image is landscape, rotate it to portrait mode'
    orig = np.swapaxes(orig, 0, 1)
    orig = orig[:, ::-1, :]

# Resize the image to speed up the processing
new_height = 1250
ratio = float(orig.shape[0]) / new_height
new_width = int(orig.shape[1] / ratio)
img = cv2.resize(orig,(new_width, new_height), interpolation = cv2.INTER_CUBIC)

# Preprocess the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blured = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(blured, 30, 200)

# Find contours in the edged image, keep only the largest ones, and initialize our screen contour
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

# Loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    raise NameError('No code box found.')

# Draw rectangle where a contours was found
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

# Apply the four point transform to obtain a top-down view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# Resize the image to match machine learning letter size
code_height = 200
code_ratio = float(warped.shape[0]) / code_height
code_width = int(warped.shape[1] / code_ratio)
code_image = cv2.resize(warped,(code_width, code_height), interpolation = cv2.INTER_CUBIC)

# Save extracted code block to file
cv2.imwrite( "code_image.jpg", code_image);
print 'Pairing code block successfully extracted and saved to file.'

if len(sys.argv) > 2:
    debug_mode = sys.argv[2]
else:
    debug_mode = 'False'

# Show a window with found numbers and there value
if debug_mode == 'debug':
    print 'Debug mode is on, display result in window'
    cv2.imshow('box_detected', img)
    cv2.imshow('code_image', code_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
