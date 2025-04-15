import os
import json
import glob
import yaml
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

def convert_labelme_to_yolo(json_path, output_dir, class_mapping):
    """Convert LabelMe JSON annotations to YOLO format."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_path = data.get('imagePath')
    if not image_path:
        return None
    
    image_dir = os.path.dirname(json_path)
    image_path = os.path.join(image_dir, image_path)
    
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None
    
    # Get image dimensions
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image: {image_path}")
        return None
    
    img_height, img_width = img.shape[:2]
    
    # Create YOLO txt file
    txt_path = os.path.join(output_dir, Path(image_path).stem + '.txt')
    
    # Copy the image to the output directory
    output_img_path = os.path.join(output_dir, Path(image_path).name)
    cv2.imwrite(output_img_path, img)
    
    # Process shapes/annotations
    with open(txt_path, 'w') as f:
        for shape in data.get('shapes', []):
            label = shape.get('label')
            if label not in class_mapping:
                print(f"Warning: Unknown label {label} in {json_path}")
                continue
                
            class_id = class_mapping[label]
            points = shape.get('points', [])
            
            if shape.get('shape_type') == 'rectangle' and len(points) == 2:
                # Rectangle format: [[x1, y1], [x2, y2]]
                x1, y1 = points[0]
                x2, y2 = points[1]
            else:
                # Handle polygon points by getting min/max
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
            
            # Convert to YOLO format (normalized)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = abs(x2 - x1) / img_width
            height = abs(y2 - y1) / img_height
            
            # Write to file
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    return output_img_path

def prepare_dataset(json_dir, output_base_dir, class_names):
    """Prepare YOLO dataset from JSON annotations."""
    # Create class mapping (name -> id)
    class_mapping = {name: i for i, name in enumerate(class_names)}
    
    # Create directory structure
    os.makedirs(output_base_dir, exist_ok=True)
    train_dir = os.path.join(output_base_dir, 'train')
    val_dir = os.path.join(output_base_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return
    
    # Split into train/val (80/20)
    np.random.shuffle(json_files)
    split_idx = int(len(json_files) * 0.8)
    train_jsons = json_files[:split_idx]
    val_jsons = json_files[split_idx:]
    
    train_imgs = []
    for json_file in train_jsons:
        img_path = convert_labelme_to_yolo(json_file, train_dir, class_mapping)
        if img_path:
            train_imgs.append(img_path)
    
    val_imgs = []
    for json_file in val_jsons:
        img_path = convert_labelme_to_yolo(json_file, val_dir, class_mapping)
        if img_path:
            val_imgs.append(img_path)
    
    # Create YAML config file
    yaml_path = os.path.join(output_base_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml_data = {
            'path': os.path.abspath(output_base_dir),
            'train': os.path.join('train'),
            'val': os.path.join('val'),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        yaml.dump(yaml_data, f)
    
    print(f"Dataset prepared: {len(train_imgs)} training images, {len(val_imgs)} validation images")
    print(f"YAML config: {yaml_path}")
    return yaml_path

def train_yolov8(yaml_path, epochs=100, img_size=640, batch_size=16):
    """Train YOLOv8 model on prepared dataset."""
    # Start with a pre-trained model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=20,  # Early stopping
        save=True
    )
    
    # Validate the model
    val_results = model.val()
    print(f"Validation results: {val_results}")
    
    # Get the best weights path
    weights_dir = Path('runs/detect/train')
    best_weights = weights_dir / 'weights' / 'best.pt'
    
    print(f"Training completed. Best weights: {best_weights}")
    return best_weights

if __name__ == "__main__":
    # Define your class names here
    CLASS_NAMES = ['bag', 'intubator']
    
    # Paths for your data
    JSON_DIR = '../../dataloader/testing_data'
    OUTPUT_DIR = './dataset'
    
    # Prepare dataset
    yaml_path = prepare_dataset(JSON_DIR, OUTPUT_DIR, CLASS_NAMES)
    
    # Train model
    if yaml_path:
        best_weights = train_yolov8(yaml_path, epochs=50)