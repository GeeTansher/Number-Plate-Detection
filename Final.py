import cv2
import numpy as np

from pytesseract import pytesseract
pytesseract.tesseract_cmd = "C:\\Users\\HP\\AppData\\Local\\Tesseract-OCR\\tesseract.exe"

img = cv2.imread("NumberPlateDetection/np2.jpg")

grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# grey_img = cv2.bilateralFilter(grey_img,11,17,17)
# _,thresh = cv2.threshold(grey_img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
canny_edge = cv2.Canny(grey_img,170,200)
contours,hierarchy = cv2.findContours(canny_edge.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key = cv2.contourArea, reverse=True)[:30]

licensePlate = None

for cnt in contours:
    peri = cv2.arcLength(cnt,True)
    poly = cv2.approxPolyDP(cnt,0.01 * peri, True)
    if len(poly) == 4:   # if it is a rect
        x,y,w,h = cv2.boundingRect(cnt)
        licensePlate = grey_img[y:y+h,x:x+w]
        break

licensePlate = cv2.bilateralFilter(licensePlate,11,17,17)
hgt , wdt  = licensePlate.shape
_,thresh = cv2.threshold(licensePlate,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
text = pytesseract.image_to_string(licensePlate)
boxes = pytesseract.image_to_boxes(licensePlate)

for box in boxes.splitlines():
    box = box.split()
    x,y,w,h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
    cv2.rectangle(licensePlate,(x,hgt-y),(w,hgt-h), (0,0,255), 1)
    cv2.putText(licensePlate,box[0],(x,hgt-h), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
    
    
print(text)
cv2.imshow("image",licensePlate)
cv2.waitKey()
cv2.destroyAllWindows()