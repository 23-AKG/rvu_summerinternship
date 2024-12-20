import torch
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Akarsh Kumar Gowda/AKG/AKG FILES/RVU stuff/Summer/yolov5/runs/train/exp3/weights/best.pt')

# Set the path to your test images
test_images_path = Path('C:/Users/Akarsh Kumar Gowda/AKG/AKG FILES/RVU stuff/Summer/test')
output_path = Path('C:/Users/Akarsh Kumar Gowda/AKG/AKG FILES/RVU stuff/Summer/test_results-3')


output_path.mkdir(parents=True, exist_ok=True)

# Function to count and visualize sunspots in an image
def process_sunspots(image_path):
    # Read the image
    img = cv2.imread(str(image_path))
    original_img = img.copy()
    
    # Run YOLOv5 inference on the image
    results = model(img)
    
    # Get the number of detected sunspots
    num_sunspots = len(results.xyxy[0])
    
    # Sort detections by x-coordinate to help with color alternation
    sorted_detections = sorted(results.xyxy[0], key=lambda x: x[0])
    
    # Define two colors for alternating
    colors = [(0, 255, 0), (255, 0, 0)]  # Green and Blue
    
    for i, detection in enumerate(sorted_detections):
        x1, y1, x2, y2, conf, _ = detection.tolist()
        color = colors[i % 2]  # Alternate colors
        
        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Calculate position for confidence text
        text = f'Conf: {conf:.2f}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Adjust text position if it would go off the top of the image
        if y1 - text_size[1] - 5 < 0:
            text_y = y1 + text_size[1] + 5
        else:
            text_y = y1 - 5
        
        # Draw confidence text
        cv2.putText(img, text, (int(x1), int(text_y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add text with total sunspot count
    cv2.putText(img, f'Total Sunspots: {num_sunspots}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Create a side-by-side comparison
    comparison = np.hstack((original_img, img))
    
    # Display the comparison
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    plt.title(f"Image: {image_path.name}, Sunspots detected: {num_sunspots}")
    plt.axis('off')
    plt.show()
    
    return num_sunspots, comparison

# Process all images in the test folder
total_images = 0
total_sunspots = 0

for img_path in tqdm(list(test_images_path.glob('*.jpg'))):  # Adjust file extension if needed
    sunspots, comparison = process_sunspots(img_path)
    total_sunspots += sunspots
    total_images += 1
    
    # Save the comparison image
    output_file = output_path / f'{img_path.stem}_comparison.jpg'
    cv2.imwrite(str(output_file), comparison)
    
    print(f"Image: {img_path.name}, Sunspots detected: {sunspots}")
    
    # Wait for user input to continue
    input("Press Enter to continue to the next image...")

# Print summary
print(f"\nTotal images processed: {total_images}")
print(f"Total sunspots detected: {total_sunspots}")
print(f"Average sunspots per image: {total_sunspots / total_images:.2f}")
print(f"\nProcessed images saved in: {output_path}")