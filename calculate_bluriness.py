import numpy as np
import cv2
import math

#capturing video frame and applying background subtraction
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
ret, frame = cap.read()
fgmask = fgbg.apply(frame)
kernel = np.ones((5, 5), np.uint8)


#calculating laplacian variation
def laplacian_variation(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return score

def calculate_bluriness(src):
    Gx = cv2.Sobel(src,cv2.CV_64F,1,0)
    Gy = cv2.Sobel(src,cv2.CV_64F,0,1)
    normGx = cv2.norm(Gx)
    normGy = cv2.norm(Gy)
    sumSq = normGx * normGx + normGy * normGy
    return 1. / (sumSq / src.shape[0]*src.shape[1] + math.exp(-6))


while (True):
    ret, frame = cap.read()
    if (frame is None):
        print("camera is blocked")
        break
    else:
        frame = cv2.resize(frame,(500,500))
        print("calculate bluriness: ",calculate_bluriness(frame))
        fm = laplacian_variation(frame)
        print("fm: ",fm)
        a = 0
        bounding_rect = []
        fgmask = fgbg.apply(frame)
        cv2.imshow("fgmask",fgmask)
        fgmask = cv2.erode(fgmask, kernel, iterations=5)
        fgmask = cv2.dilate(fgmask, kernel, iterations=5)
        cv2.imshow('fgmask 1', fgmask)


        #taking threshold, calling laplacian_variation and calculating
        #contours
        ret, thresh = cv2.threshold(frame, 10, 50, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(0, len(contours)):
            bounding_rect.append(cv2.boundingRect(contours[i]))

        for i in range(0, len(contours)):
            if bounding_rect[i][2] >= 40 or bounding_rect[i][3] >= 40:
                a = a + (bounding_rect[i][2]) * bounding_rect[i][3]
                #tinh tong dien tich cac hinh bao

            #check if focus measure(fm) is less than threshold then
            #camera is out out of focus or incoming images is blurry
            if fm < thresh.all() or a >= int(frame.shape[0]) * int(frame.shape[1]) / 2:
                cv2.putText(frame, "defocused", (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,
                            255), 2)
                cv2.putText(frame, "{}: {:.2f}".format('blurriness',
                            fm), (5, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255),3)

        cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

#release the camera and destroy all opened windows
cap.release()
cv2.destroyAllWindows()
