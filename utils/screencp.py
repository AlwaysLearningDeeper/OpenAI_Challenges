import numpy as np
from PIL import ImageGrab
from utils.directkeys import ReleaseKey, PressKey, W, A, S, D
import cv2
import time
from threading import Thread





def process_img(original_image):
    processed_img= cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    processed_img=cv2.Canny(processed_img, threshold1=200,threshold2=300)
    return processed_img

def showscreen():
    print("showscreen enabled")
    while(True):
        screen= np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        new_screen=process_img(screen)
        #printscreen_numpy =   np.array(printscreen_pil.getdata(),dtype='uint8')\
        last_time=time.time()
        #print('Loop took  seconds'+str(1/(time.time()-last_time)))
        cv2.imshow('window',new_screen)
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
def move():
    print("Starting to move in 5 seconds")
    print(5)
    time.sleep(1)
    print(4)
    time.sleep(1)
    print(3)
    time.sleep(1)
    print(2)
    time.sleep(1)
    print(1)
    time.sleep(1)
    print("Starting")
    while(True):
        PressKey(D)
        time.sleep(1)
        ReleaseKey(D)

        PressKey(A)
        time.sleep(1)
        ReleaseKey(A)

if __name__ == "__main__":
    t1 = Thread(target = showscreen)
    t2 = Thread(target = move)
    t1.setDaemon(True)
    t2.setDaemon(True)
    t1.start()
    t2.start()
    while True:
        pass
