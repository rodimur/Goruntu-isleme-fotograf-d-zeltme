import cv2 as cv
import numpy as np

img = cv.imread("resim.jpeg")
cv.imshow("Orijinal Resim", img)

markedDamages = cv.imread("resim.jpeg", 0)
ret, thresh = cv.threshold(markedDamages, 215, 300, cv.THRESH_BINARY)
cv.imshow("Mask Threshold", thresh)

kernel = np.ones((7, 7), np.uint8)
mask = cv.dilate(thresh, kernel, iterations=1)
cv.imshow("Mask Processed", mask)

restoredImage = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)
cv.imshow("Restored", restoredImage)

cv.waitKey(0)
cv.destroyAllWindows()