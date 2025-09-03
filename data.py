import os
import cv2
import yaml

# --- Configuration ---
# Replace with the path to the folder you downloaded from Roboflow
# Use forward slashes to avoid syntax errors
dataset_path = "C:/Users/white/Downloads/Data"

# Create directories for the new datasets
output_healthy_dir = "healthy_images"
output_unhealthy_dir = "unhealthy_images"
os.makedirs(output_healthy_dir, exist_ok=True)
os.makedirs(output_unhealthy_dir, exist_ok=True)

# --- Main Logic ---

# Load class names from the data.yaml file to map class indices to names
try:
    with open(os.path.join(dataset_path, "data.yaml"), 'r') as file:
        data = yaml.safe_load(file)
        class_names = data['names']
        class_map = {name: i for i, name in enumerate(class_names)}
        
    healthy_class_index = class_map.get('Healthy Leaf')
    unhealthy_class_index = class_map.get('Unhealthy Leaf')
    
    if healthy_class_index is None or unhealthy_class_index is None:
        raise ValueError("Class names 'Healthy Leaf' or 'Unhealthy Leaf' not found in data.yaml.")
except FileNotFoundError:
    print("Warning: data.yaml not found. Manually set class indices.")
    healthy_class_index = 0
    unhealthy_class_index = 1
except Exception as e:
    print(f"An error occurred while reading data.yaml: {e}")
    exit()

# Process each split (train, valid, test)
for split in ["train", "valid", "test"]:
    image_dir = os.path.join(dataset_path, split, "images")
    label_dir = os.path.join(dataset_path, split, "labels")

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"Skipping {split} split, as the folder does not exist.")
        continue

    print(f"Processing {split} split...")
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            image_file = label_file.replace(".txt", ".jpg")  # Adjust extension if needed (.png)
            image_path = os.path.join(image_dir, image_file)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found for {label_file}. Skipping.")
                continue
                
            img = cv2.imread(image_path)
            h, w, _ = img.shape
            
            with open(os.path.join(label_dir, label_file), "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    
                    try:
                        parts = line.strip().split()
                        
                        # The core check: if the line doesn't have 5 parts, it's malformed.
                        if len(parts) != 5:
                            print(f"Skipping malformed line in file: {label_file}, line number: {i+1}. Expected 5 values, got {len(parts)}.")
                            continue

                        class_id = int(parts[0])
                        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
                        
                        # Convert normalized YOLO format to pixel coordinates
                        x_min = int((x_center - bbox_width/2) * w)
                        y_min = int((y_center - bbox_height/2) * h)
                        x_max = int((x_center + bbox_width/2) * w)
                        y_max = int((y_center + bbox_height/2) * h)
                        
                        # Ensure coordinates are within image bounds
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(w, x_max)
                        y_max = min(h, y_max)
                        
                        cropped_image = img[y_min:y_max, x_min:x_max]
                        
                        # Save the cropped image to the correct folder
                        if class_id == healthy_class_index:
                            output_path = os.path.join(output_healthy_dir, f"{os.path.basename(image_file).replace('.jpg', '')}_crop{i}.jpg")
                            cv2.imwrite(output_path, cropped_image)
                        elif class_id == unhealthy_class_index:
                            output_path = os.path.join(output_unhealthy_dir, f"{os.path.basename(image_file).replace('.jpg', '')}_crop{i}.jpg")
                            cv2.imwrite(output_path, cropped_image)
                    
                    except (ValueError, IndexError) as e:
                        # This catches any other parsing errors on a specific line
                        print(f"Skipping line due to an error in file: {label_file}, line number: {i+1}")
                        print(f"Error message: {e}")
                        continue

print("\nCropping and separation complete!")