import numpy as np
from utils.grabscreen import grab_screen
import cv2
import time
import os



def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(title='FCEUX 2.2.2: Castlevania (USA)')
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            #screen = cv2.resize(screen, (160,90))
            cv2.imshow('window',screen)
            #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
main()
