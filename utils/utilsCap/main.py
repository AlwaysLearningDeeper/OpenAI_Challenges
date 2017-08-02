import numpy as np
from Project.utils.grabscreen import grab_screen
from matplotlib import pyplot as plt
from Project.utils.utilsCap.grabkeys import key_check
import cv2
import time
import sys
import os


def keys_to_output(keys):
    #[A,D]
    output=[0,0]
    if 'A' in keys:
        output[0]=1
    elif 'D' in keys:
        output[1]=1
    return output

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading data')
    training_data=list(np.load(file_name))
else:
    print('File does not exists, creating one')
    training_data=[]


def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    c=0
    last_time = time.time()
    while True:
        c+=1
        screen=grab_screen(title='')
        screenG=cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
        screenG=cv2.resize(screenG,(80,60))
        keys=key_check()
        output=keys_to_output(keys)
        training_data.append([screenG,output])
        if c%10==0:
            print('Recording at ' + str((10 / (time.time() - last_time)))+' fps')
            last_time = time.time()

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name,training_data)
