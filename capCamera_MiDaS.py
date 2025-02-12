import cv2
import numpy as np
import torch

def init_MiDaS(model_type = "MiDaS_small"):
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    device = torch.device("cuda")
    
    return model, transform, device

def init_Zoe():
    pass

def display_MiDaS(torch_mod = None):
    if torch_mod is None:
        model_type = "MiDaS_small"
        model, transform, device = init_MiDaS("MiDaS_small")
    else:
        model, transform, device = torch_mod
    model.to(device)
    model.eval()
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    
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
    
    scale = 0.25
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Error: Could not read frame.")
            break
        frame = cv2.flip(frame, 1)
        frame_r = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        
        # -------------------------------------------------------------------------------------------------------------------------------------
        
        img = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)
        
        with torch.no_grad():
            prediction = model(input_batch)
            
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # -------------------------------------------------------------------------------------------------------------------------------------
        
        depth_map = prediction.cpu().numpy()
        normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        normalized_depth_map_color = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_BONE)
        depth_frame = cv2.resize(normalized_depth_map_color, (width,height))
        
        cv2.imshow("Depth Map", depth_frame)
        
        # -------------------------------------------------------------------------------------------------------------------------------------
        
        key = cv2.waitKey(1)  # Wait for 1ms. Press 'q' to quit.
        if key == ord('q'):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows

def minth():
    display_MiDaS()

if __name__ == "__main__":
    minth()
