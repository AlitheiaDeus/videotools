import cv2
import os
import glob
import time

def chooseVid():
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = []
    
    for ext in supported_formats:
        video_files.extend(glob.glob('*'+ext))
    
    if not video_files:
        print("No supported video files found in the current directory.")
        return None
    
    print("Select a video file:")
    for i, filename in enumerate(video_files):
        print(f"{i + 1}. {filename}")
    
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(video_files):
                return video_files[choice - 1]
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def streamVid(filename, windowWidth, windowHeight):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {filename}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:  # Handle cases where FPS cannot be determined
        fps = 30 # A default
    delay = 1 / fps
    
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', windowWidth, windowHeight)
    cv2.moveWindow('Video', 0, 0)
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Get original frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Calculate scaling factors
        scale_width = windowWidth / frame_width
        scale_height = windowHeight / frame_height
        scale = min(scale_width, scale_height)  # Choose the smaller scale to maintain aspect ratio
        
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        resized_frame = cv2.resize(frame, (0, 0), fx = scale, fy = scale)
        
        
        cv2.imshow('Video', resized_frame)
        
        # elapsed_time = (time.time() - start_time) * 1000
        # wait_time = max(1, delay - int(elapsed_time))
        
        while time.time() < start_time + delay:
            pass
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Adjust waitKey value for playback speed
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("In the beninging")
    
    windowWidth, windowHeight = 1280, 720
    
    filename = chooseVid()
    
    
    
    streamVid(filename, windowWidth, windowHeight)

if __name__ == "__main__":
    main()