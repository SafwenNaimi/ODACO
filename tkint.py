import cv2


with open('test.txt', 'r') as f:
    a=f.read()

img=cv2.imread('vira.jpg')

cv2.imshow('imgae',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.waitKey(1)

