import cv2
import numpy as np
from PIL import ImageGrab

def display_screen():
    scale = 0.25
    
    while True:
        screenshot = ImageGrab.grab()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        
        cv2.imshow("Screen Feed", frame)  # Display the frame

        key = cv2.waitKey(1)  # Wait for 1ms. Press 'q' to quit.
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()  # Close all windows

def minth():
    display_screen()

if __name__ == "__main__":
    minth()
