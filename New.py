import cv2
import numpy as np

#initializing coordinate values
x_i, y_i = 1, 1
point_selected = False

def mouse_input(event, x, y, flag, param):
    "Takes input from the mouse left click"
    global x_i, y_i, point_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Point is selected")
        #updating the points
        x_i, y_i = x, y
        point_selected = True 

#creating event
cv2.namedWindow("Left click to track")
cv2.setMouseCallback("Left click to track", mouse_input)

#reading video from camera
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    if ret==False:
        #if no frame is detected
        break
    
    cv2.imshow("Left click to track", frame)
    k = cv2.waitKey(1)
    
    if k==27 or point_selected:
        #converting current frame to gray
        old_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.destroyAllWindows()
        break

#storing the point selected
old_pts = np.array([[x_i, y_i]], dtype="float32").reshape(-1, 1, 2)

#creating mask image to draw points 
mask_image = np.zeros_like(frame)

while video.isOpened():
    ret, current_frame = video.read()
    
    if ret==False:
        #if no frame is detected
        break
    
    #convert current frame to gray
    current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    #applying Lucas Kanade method
    """
    curr_pts, status, error = cv2.calcOpticalFlowPyrLK(old_gray_frame, current_gray_frame, old_pts, 
                                                     None,
                                                     maxLevel=1,
                                                     flags=10,
                                                     minEigThreshold=0.02)
    """
    curr_pts, status, error = cv2.calcOpticalFlowPyrLK(old_gray_frame,current_gray_frame, old_pts,
                                                       None,winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
    
    print(status, error)
    #drawing lines on the image
    for (current,prev) in zip(curr_pts, old_pts):
        a,b = current.ravel()
        c,d = prev.ravel()
        mask_image = cv2.line(mask_image, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        current_frame = cv2.circle(current_frame,(int(a),int(b)), 5, (0, 255, 0), -1)
    out = cv2.addWeighted(current_frame, 1, mask_image, 0.8, 0.1)
    
    cv2.imshow("Move the point", out)
    k = cv2.waitKey(1)
    if k==27:
        break
    
    #setting the current frame as previous frame for next iteration
    prev_gray_frame = current_gray_frame.copy()
    old_pts = curr_pts.reshape(-1,1,2)
    
video.release()
cv2.destroyAllWindows()