import numpy as np
from Project.utils.grabscreen import grab_screen
from Project.utils.directkeys import PressKey,ReleaseKey,A,W,S,D
import cv2
import time
import sys
import os

def draw_lines(img,lines):
    try:
        for l in lines:
            coords=l[0]
            cv2.line(img,(coords[0],coords[1]),(coords[2],coords[3]),[230,230,230],3)
    except:
        pass

def draw_circles(img,circles):
    try:
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    except:
        pass
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

def calmdown():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def roi(img,vertices):
    # blank mask:
    mask = np.zeros_like(img)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, 255)

    # returning the image only where mask pixels are nonzero
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    processed_img = cv2.GaussianBlur(processed_img, (3,3), 0 )
    vertices = np.array([[10,500],[10,300], [300,200], [500,200], [800,300], [800,500]], np.int32)
    processed_img = roi(processed_img, [vertices])

    #                       edges
    #lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180,np.array([]), 120, 2)
    #draw_lines(processed_img,lines)
    circles = cv2.HoughCircles(processed_img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=75, param2=10, minRadius=3, maxRadius=5)
    draw_circles(processed_img,circles)

    return processed_img
def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    c=0
    last_time = time.time()
    while(True):
        if not paused:
            c+=1
            # 800x600 windowed mode
            screen = grab_screen(title='FCEUX 2.2.3: Arkanoid (USA)')
            if c%10==0:
                print('Recording at ' + str((10 / (time.time() - last_time)))+' fps')
                last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            height, width, channels = screen.shape
            vertices=np.array([[10,270],[10,100],[150,100],[150,270]])
            print(height)
            print(width)
            screen=roi(screen,vertices)
            screen=process_img(screen)
            #screen = cv2.resize(screen, (160,90))
            cv2.imshow('window',screen)
            #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
main()
