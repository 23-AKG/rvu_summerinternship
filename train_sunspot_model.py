import torch
import subprocess
import sys
import os

# Set up configuration
data_yaml = 'C:/Users/Akarsh Kumar Gowda/AKG/AKG FILES/RVU stuff/Summer/data_split/dataset.yaml'
weights = 'yolov5n.pt'  # or 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt'
epochs = 5  
img_size = 640  # Increased from 320 to 640 for better detection

# Run the training
if __name__ == '__main__':
    # Ensure we're in the correct directory
    yolov5_dir = 'C:/Users/Akarsh Kumar Gowda/AKG/AKG FILES/RVU stuff/Summer/yolov5'  # Replace with the actual path to your YOLOv5 directory
    os.chdir(yolov5_dir)

    # Download the pre-trained weights
    torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
    
    # Construct the command
    command = [
        sys.executable,
        'train.py',
        '--img', str(img_size),
        '--batch', '16',
        '--epochs', str(epochs),
        '--data', data_yaml,
        '--weights', weights,
        '--cache'  # Added cache for faster training
    ]
    
    # Run the command
    subprocess.run(command, check=True)