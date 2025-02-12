import cv2

import time
import os

import torch
import torchvision.transforms as T

import numpy as np

from PIL import Image

from pytubefix import YouTube
from pytubefix.cli import on_progress

def init_ZoeD():
    # INIT TORCH
    model_type = "ZoeD_NK"
    model_zoe_nk = torch.hub.load("isl-org/ZoeDepth", model_type, pretrained=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("WHERE TF IS CUDA?!")
        return False
    model_zoe_nk.to(device)
    model_zoe_nk.eval()
    transform = T.ToTensor()
    
    # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    # transform = midas_transforms.small_transform
    
    return transform, device, model

def ZoeDify(image, frame_height, frame_width, transform, device, model):    
    input_batch = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
    
    depth_map = prediction.squeeze().cpu().numpy()
    normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_frame = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_BONE)
    
    return depth_frame

def get_url():
    url = input("Paste your link here: ")
    if url:
        return url
    else:
        return None
    
def read_youtube(url):
    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        print(f"Found stream: {yt.title}")
        stream = yt.streams.get_highest_resolution()
        
        filepath = stream.default_filename
        
        if os.path.exists(filepath):
            print(f"Video '{yt.title}' already exists at: {filepath}")
            return filepath  # Return existing filepath
        
        print(f"Downloading '{yt.title}'")
        filepath = stream.download()
        
    except Exception as e:
        print()
        print(e)
        print()
        return None
    
    return filepath

def stream_mp4(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"Error opening {filepath}")
        return None
    
    frame_width = int(cap.get(3))   # Get frame width
    frame_height = int(cap.get(4))  # Get frame height
    fps = cap.get(cv2.CAP_PROP_FPS) # Get FPS
    if fps == 0:
        fps = 30
        print("Warning: FPS not detected. Using default FPS of 30.")
    else:
        print(f"Playing at FPS of {fps}.")
    
    target_delay = 1/fps
    
    transform, device, model = init_ZoeD()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break #Break the loop if you can't get a frame
        else:
            frame_s = frame.copy()
            frame_rgb = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB) # Important for PIL to work correctly
            pil_image = Image.fromarray(frame_rgb)
        
        # print(frame_s)
        # new_height, new_width, _ = frame_s.size
        
        depth_frame = ZoeDify(pil_image, frame_height, frame_width, transform, device, model)
        
        gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        low_threshold = 50
        high_threshold = 150
        canny_depth = cv2.Canny(blurred, low_threshold, high_threshold)
        canny_depth_frame = cv2.cvtColor(canny_depth, cv2.COLOR_GRAY2BGR)
        
        stacked = np.concatenate((frame_s, depth_frame, canny_depth_frame), 1)
        
        cv2.imshow("Frame", stacked)
        # cv2.imshow("Video", frame_s)
        # cv2.imshow("Depth", depth_frame)
        # cv2.imshow("Canny", canny_depth_frame)
        
        current_time = time.time()
        elapsed_time = current_time - (getattr(stream_mp4, 'last_frame_time', 0))
        stream_mp4.last_frame_time = current_time
        
        delay = target_delay - elapsed_time
        if delay > 0:
            time.sleep(delay)
        
        key = cv2.waitKey(1) # Wait for 1ms. Press 'q' to quit.
        if key == ord('q'):
            break
    
    cap.release()           # Release the capture
    # out.release()           # Release the writer
    cv2.destroyAllWindows() # Close all windows
    return None

def main():
    start_time = time.time()
    
    url = get_url()
    if url == None:
        return
    
    filepath = read_youtube(url)
    if filepath == None:
        return
    
    q = stream_mp4(filepath)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nExited with:\n{q}\nat {elapsed_time}")
    
    return

if __name__ == "__main__":
    main()