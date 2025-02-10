def display_webcam_native_resolution(loop_chk=True, rec_chk=False):
    import cv2
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Attempt to set to 4K or max cam size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam Native Resolution: {width} x {height}")
    
    cv2.namedWindow("Webcam Feed", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Webcam Feed", 0,0)
    
    if not loop_chk:
        ret, frame = cap.read()
        cap.release()
        frame = cv2.flip(frame, 1)
        if not ret:
            print("Error: Could not read frame.")
            return None
        cv2.imshow("Webcam Feed", frame)
        
        return frame
    
    if rec_chk:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter("output.mp4", fourcc, 30, (width, height))
    
    print("Beginning Feed. Press 'q' to exit.")
    
    import time
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        cv2.imshow("Webcam Feed", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        if rec_chk:
            out.write(frame)
    
    cap.release()
    if rec_chk:
        out.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    print(f"Streaming time: {end_time - start_time}s")

def minth():
    display_webcam_native_resolution()
    return None
if __name__ == "__main__":
    minth()