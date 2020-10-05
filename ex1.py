import cv2


img=cv2.imread('start.png')
#cv2.imshow('pdgd',img)
cv2.namedWindow ('key', cv2.WINDOW_NORMAL)
cv2.setWindowProperty ('key', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('key',img)
cv2.waitKey(0)
with open('testing.txt', 'r') as f:
    a=f.read()
if a=='do stop':
    cv2.destroyAllWindows()

