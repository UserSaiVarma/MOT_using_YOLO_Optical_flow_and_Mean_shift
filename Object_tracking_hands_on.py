import cv2
import numpy as np
from Object_detection_hands_on import ObjectDetection

#Loading object detection
od = ObjectDetection()
class_names = od.load_class_names()
center_pts = []

video  = cv2.VideoCapture("los_angeles.mp4")
while True:
    ret, frame = video.read()
    if ret==False:
        #No frame is detected
        print("Completed")
        break
    
    (class_ids, scores, boxes) = od.detect(frame)
    
    #displaying results
    for idx, box in enumerate(boxes):
        (x, y, w, h) = box
        #taking centers
        (cx, cy) = (int(x+w/2), int(y+h/2))
        center_pts.append((cx, cy))
        
        #drawing bounding box
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        # Using cv2.putText() method
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "{}-{}".format(class_names[class_ids[idx]], str(round(scores[idx],2)))
        cv2.putText(frame, text, (x,y), font, 0.5, (150,0,0), 1, cv2.LINE_AA)
    
    for pt in center_pts:
        cv2.circle(frame, pt, 5, (0,0,255), -1)
    
    #Ouput images
    cv2.imshow("video", frame)
    k = cv2.waitKey(1)
    if k==27:
        print("Interrupted")
        break
    
    
video.release()
cv2.destroyAllWindows()
print("end")