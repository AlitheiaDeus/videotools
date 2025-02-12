# Uses sequential frames on a mono camera to estimate stereo depth in the horizontal direction

import cv2
import numpy as np

def display_depthcam():
    def compute_disparity(frameL, frameR, stereo):
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        
        disparity = stereo.compute(frameL, frameR).astype(np.float32) / 16.0
        disparity_frame = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return disparity_frame
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Attempt to set to 4K or max cam size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam Native Resolution: {width} x {height}")
    
    cv2.namedWindow("Depth Map", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Depth Map", 0,0)
    
    previous_frame_r = None
    
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=4)
    
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Error: Could not read frame.")
            break
        frame = cv2.flip(frame, 1)
        frame_r = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        
        if previous_frame_r is not None:
            disparity1_visual = compute_disparity(frame_r, previous_frame_r, stereo)
            disparity2_visual = compute_disparity(previous_frame_r, frame_r, stereo)
            
            disparity_visual = cv2.addWeighted(disparity1_visual, 0.5, disparity2_visual, 0.5, 0)
            disparity_visual = cv2.applyColorMap(disparity_visual, cv2.COLORMAP_BONE)
            disparity_visual = cv2.resize(disparity_visual, (0,0), fx=4, fy=4)
            
            # overlay = cv2.addWeighted(frame, 0.7, disparity_visual, 0.7, 0)
            # cv2.imshow("Depth Map", overlay)
            
            cv2.imshow("Depth Map", disparity_visual)
        
        previous_frame_r = frame_r.copy()
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return None

def minth():
    display_depthcam()
    return
if __name__ == "__main__":
    minth()
