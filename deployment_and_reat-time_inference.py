import jetson.inference
import jetson.utils
import cv2
import numpy as np 
import time

width= 1280
height= 720
cam1= cv2.VideoCapture('/dev/video0')
cam1.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
#net=jetson.inference.detectNet('ssd-mobilenet-v2',threshold=0.5)
net=jetson.inference.detectNet(argv=["--model=/home/gdp-4/Downloads/jetson-inference/python/training/detection/ssd/models/myModel/ssd-mobilenet.onnx","--labels=/home/gdp-4/Downloads/jetson-inference/python/training/detection/ssd/models/myModel/labels.txt","--input-blob=input_0","--output-cvg=scores","--output-bbox=boxes"],threshold=0.8)
font= cv2.FONT_HERSHEY_SIMPLEX
timeMark=time.time()
fpsFilter=0

is_ha=0
is_ha2=0
is_ha3=0
is_ha4=0
endtime=-1
ha_time=-1
chest_time1=90000000000
chest_time2=-1
falltime=90000000000

while True:
    
    _,img= cam1.read()
    h= img.shape[0]
    w= img.shape[1]
    frame= cv2.cvtColor(img,cv2.COLOR_BGR2RGBA).astype(np.float32)
    frame= jetson.utils.cudaFromNumpy(frame)
    detections= net.Detect(frame,w,h)
    
    for detect in detections:
        ID= detect.ClassID
        top= int(detect.Top)
        left= int(detect.Left)
        bottom= int(detect.Bottom)
        right= int(detect.Right)
        item= net.GetClassDesc(ID)

        if (item=='chest-pain')&(is_ha==0):
            print('inside')
            is_ha=1
            endtime=time.time()+20
            chest_time1=time.time()+15
            chest_time2=time.time()+30

        if ((item=='fall')&(is_ha==1)&(is_ha3==0)&(time.time()<endtime)):
           is_ha3=1
           is_ha=0
           falltime=time.time()+10
        if ((item=='fall')&(is_ha3==1)&(time.time()>=endtime)):
            is_ha2=1
            is_ha3=0
            falltime=90000000

        if ((item=='chest-pain')&(is_ha==1)&(time.time()>=chest_time1)):
            if (time.time()<chest_time2):
                is_ha2=1
                is_ha=0
                chest_time1=900000000
                chest_time2=-1
            else:
                is_ha=0
                chest_time1=900000000
                chest_time2=-1
            #is_ha2=1
            #is_ha=0
            #print(is_ha)
            #print(is_ha2)
            #chest_time1=900000000
            #chest_time2=-1

        conf= detect.Confidence
        cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),1)
        cv2.putText(img,item +' '+str(round(conf,2)),(left,top+20),font,.75,(0,0,255),2)

    dt=time.time()-timeMark
    fps=1/dt
    fpsFilter=.9*fpsFilter + .1*fps
    timeMark=time.time()
    timer=timeMark-16474779
    if is_ha2==1:
        is_ha2=0
        ha_time=time.time()+3
        cv2.putText(img,str(round(fpsFilter,1))+' FPS '+' Possible Heart Attack!!! ',(0,30),font,1,(0,0,255),2)
        cv2.imshow('recCam',img)
        cv2.moveWindow('recCam',0,0)
    elif time.time()<=ha_time:
        cv2.putText(img,str(round(fpsFilter,1))+' FPS '+' Possible Heart Attack!!! ',(0,30),font,1,(0,0,255),2)
        cv2.imshow('recCam',img)
        cv2.moveWindow('recCam',0,0)
    else:
        cv2.putText(img,str(round(fpsFilter,1))+' FPS ',(0,30),font,1,(0,0,255),2)
        cv2.imshow('recCam',img)
        cv2.moveWindow('recCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()       
cv2.destroyAllWindows()        