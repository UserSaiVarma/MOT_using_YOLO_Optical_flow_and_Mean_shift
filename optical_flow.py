#importing the libraries
import cv2 
import numpy as np

class Optical_FLow:
    def __init__(self, cl=None):
        pass
    
    def get_features(self, gray_image):
        "Detecting features(corners) using Shi-Thomasi Corner Detectors"
        points = cv2.goodFeaturesToTrack(gray_image, mask=None, maxCorners=100,
                                         qualityLevel=0.1, minDistance=10, blockSize=5)
        return points
    
    def LK_model(self, gray_t_1, gray_t, points_t_1):
        "Detecting Optical Flow using Lucas Kanade Sparse Optical FLow method"
        points, status, error = cv2.calcOpticalFlowPyrLK(gray_t_1,gray_t,points_t_1,None,
                                                         winSize=(10,10),maxLevel=2, flags=10, minEigThreshold=0.03)
        return points, status, error
    
    def draw_flow(self, current_frame, prev_gray_frame, prev_points, mask_image):
        #coloring the flow lines
        color = np.random.randint(0,255,(100,3))
        
        current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_points, status, error = self.LK_model(prev_gray_frame, current_gray_frame, prev_points)
        
        if current_points is not None:
            current_points = current_points[status==1]
            old_points = prev_points[status==1]
        
        for i,(current,prev) in enumerate(zip(current_points, old_points)):
            a,b = current.ravel()
            c,d = prev.ravel()
            mask_image = cv2.line(mask_image, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            current_frame = cv2.circle(current_frame,(int(a),int(b)), 5, color[i].tolist(), -1)
        #out = cv2.add(current_frame, mask_image)
        out = mask_image
        
        #setting the current frame as previous frame for next iteration
        prev_gray_frame = current_gray_frame.copy()
        prev_points = current_points.reshape(-1,1,2)
        
        return out, prev_gray_frame, prev_points, mask_image