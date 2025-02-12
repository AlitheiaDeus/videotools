import cv2
import numpy as np

def display_flowcam():
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
    
    scale = 0.5
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Could not read frame.")
        cap.release()  # Release the camera
        return
    frame = cv2.flip(frame, 1)
    frame_r = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    
    height_r, width_r = frame_r.shape[:2]
    
    prev_gray = None
    
    threshold = 2.0  # Adjust this threshold value
    
    cv2.namedWindow("Flow Map", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Flow Map", 0,0)
    
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Error: Could not read frame.")
            break
        frame = cv2.flip(frame, 1)
        frame_r = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        
        flow_im = np.zeros_like(frame_r)
        
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_r, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # Adjust parameters
            
            for y in range(0, height_r, 10):  # Sample flow vectors for display
                for x in range(0, width_r, 10):
                    fx, fy = flow[y, x]
                    
                    magnitude = np.sqrt(fx**2 + fy**2) # Calculate vector magnitude
                    
                    if magnitude > threshold:
                        cv2.line(flow_im, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1)
        
        flow_im = cv2.resize(flow_im, (width, height))
        
        overlay = cv2.addWeighted(flow_im, 0.2, frame, 0.8, 0)
        
        cv2.imshow("Flow Map", overlay)  # Display the frame
        
        prev_gray = gray_r.copy() # Update previous frame
        
        key = cv2.waitKey(1)  # Wait for 1ms. Press 'q' to quit.
        if key == ord('q'):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows

if __name__ == "__main__":
    display_flowcam()
