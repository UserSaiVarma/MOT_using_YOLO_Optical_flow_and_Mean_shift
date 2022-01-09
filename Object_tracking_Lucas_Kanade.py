import cv2
import numpy as np
from Object_detection_hands_on import ObjectDetection
from optical_flow import Optical_FLow

#Loading object detection
od = ObjectDetection()
class_names = od.load_class_names()

#Loading optical flow
of = Optical_FLow()

#Frame count
count = 0

video  = cv2.VideoCapture("los_angeles.mp4")
while True:
    ret, frame = video.read()
    if ret==False:
        #No frame is detected
        break
    
    if count==0:
        #Reading frame adn finding features
        prev_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_points = of.get_features(prev_gray_frame)
        mask_image = np.zeros_like(frame)
    
    of_frame = frame
    if count!=0:
        (of_frame, prev_gray_frame, prev_points, mask_image) = of.draw_flow(frame, prev_gray_frame, prev_points, mask_image)

    (class_ids, scores, boxes) = od.detect(frame)
    
    #displaying results
    for idx, box in enumerate(boxes):
        (x, y, w, h) = box
        #drawing bounding box
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        # Using cv2.putText() method
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "{}-{}".format(class_names[class_ids[idx]], str(round(scores[idx],2)))
        cv2.putText(frame, text, (x,y), font, 0.5, (150,0,0), 1, cv2.LINE_AA)
    count += 1
    #Ouput images
    out = cv2.add(of_frame, frame)
    cv2.imshow("video", out)
    k = cv2.waitKey(1)
    if k==27:
        break
    
    
video.release()
cv2.destroyAllWindows()
print("end")