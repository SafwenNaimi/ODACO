import cv2
import numpy as np     # Numpy module will be used for horizontal stacking of two frames

video=cv2.VideoCapture(0)
a=0
while True:
    a=a+1
    check, frame= video.read()

    # Converting the input frame to grayscale
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   

    # Fliping the image as said in question
    gray_flip = cv2.flip(gray,1)

    # Combining the two different image frames in one window
    combined_window = np.hstack([gray,gray_flip])

    # Displaying the single window
    cv2.imshow("Combined videos ",combined_window)
    key=cv2.waitKey(1)

    if key==ord('q'):
        break
print(a)

video.release()
cv2.destroyAllWindows
