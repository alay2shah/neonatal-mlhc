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
    """Prepare YOLO dataset with additional augmentations for better recall."""
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
    
    # Process training data
    train_imgs = []
    for json_file in train_jsons:
        img_path = convert_labelme_to_yolo(json_file, train_dir, class_mapping)
        if img_path:
            train_imgs.append(img_path)
            
            # Add basic augmentations to training set to improve recall
            img = cv2.imread(img_path)
            if img is not None:
                # Horizontal flip augmentation
                img_flipped = cv2.flip(img, 1)
                flip_path = os.path.join(train_dir, f"{Path(img_path).stem}_flipped{Path(img_path).suffix}")
                cv2.imwrite(flip_path, img_flipped)
                
                # Create flipped annotation file
                txt_path = os.path.join(train_dir, f"{Path(img_path).stem}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        annotations = f.readlines()
                    
                    flip_txt_path = os.path.join(train_dir, f"{Path(img_path).stem}_flipped.txt")
                    with open(flip_txt_path, 'w') as f:
                        for ann in annotations:
                            parts = ann.strip().split()
                            if len(parts) >= 5:
                                # Flip x-coordinate (1.0 - x_center)
                                parts[1] = str(1.0 - float(parts[1]))
                                f.write(' '.join(parts) + '\n')
                
                # Brightness augmentation
                brightness = np.random.uniform(0.8, 1.2)
                img_bright = np.clip(img * brightness, 0, 255).astype(np.uint8)
                bright_path = os.path.join(train_dir, f"{Path(img_path).stem}_bright{Path(img_path).suffix}")
                cv2.imwrite(bright_path, img_bright)
                
                # Copy same annotation for brightness augmentation
                bright_txt_path = os.path.join(train_dir, f"{Path(img_path).stem}_bright.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as src, open(bright_txt_path, 'w') as dst:
                        dst.write(src.read())
    
    # Process validation data
    val_imgs = []
    for json_file in val_jsons:
        img_path = convert_labelme_to_yolo(json_file, val_dir, class_mapping)
        if img_path:
            val_imgs.append(img_path)
    
    # Create YAML config file with expanded datasets
    yaml_path = os.path.join(output_base_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml_data = {
            'path': os.path.abspath(output_base_dir),
            'train': os.path.join('train'),
            'val': os.path.join('val'),
            'names': {i: name for i, name in enumerate(class_names)},
            'nc': len(class_names)  # Add explicit number of classes
        }
        yaml.dump(yaml_data, f)
    
    print(f"Dataset prepared with augmentations: {len(train_imgs)*3} training images (with augmentations), {len(val_imgs)} validation images")
    print(f"YAML config: {yaml_path}")
    return yaml_path
def train_yolov8(yaml_path, epochs=100, img_size=640, batch_size=16):
    """Train YOLOv8 model with improved data augmentation and no mosaic."""
    # Start with a pre-trained model
    model = YOLO('yolov8n.pt')
    
    # Train the model with optimized hyperparameters for better recall
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=20,  # Early stopping
        save=True,
        
        # Disable mosaic augmentation
        mosaic=0.0,  # Disable mosaic completely
        
        # Improve data augmentation for better recall
        augment=True,
        degrees=10.0,  # Rotation augmentation (±10 degrees)
        translate=0.2,  # Translation augmentation (±20%)
        scale=0.2,     # Scale augmentation (±20%)
        shear=5.0,     # Shear augmentation
        flipud=0.1,    # Flip up-down probability
        fliplr=0.5,    # Flip left-right probability
        
        # Improve learning parameters for better recall
        lr0=0.01,      # Initial learning rate
        lrf=0.01,      # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        
        # Class balance improvements
        cls=0.5,       # Increase class loss weight for better detection
        box=0.05,      # Reduce box loss weight
        
        # Lower confidence threshold for training
        conf=0.1,      # Lower confidence threshold for training
        iou=0.5,       # IoU threshold
        
        # Prevent overfitting
        dropout=0.1    # Add dropout for regularization
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