import torch
from pathlib import Path
import cv2

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Akarsh Kumar Gowda/AKG/AKG FILES/RVU stuff/Summer/yolov5/runs/train/exp3/weights/best.pt')

# Set the path to your test images
test_images_path = Path('C:/Users/Akarsh Kumar Gowda/AKG/AKG FILES/RVU stuff/Summer/test')

# Function to count sunspots in an image
def count_sunspots(image_path):
    # Read the image
    img = cv2.imread(str(image_path))
    
    # Run YOLOv5 inference on the image
    results = model(img)
    
    # Get the number of detected sunspots
    num_sunspots = len(results.xyxy[0])
    
    return num_sunspots

# Process all images in the test folder
for img_path in test_images_path.glob('*.jpg'):  # Adjust file extension if needed
    sunspots = count_sunspots(img_path)
    print(f"Image: {img_path.name}, Sunspots detected: {sunspots}")