import cv2
import numpy as np
import argparse
from models import YOLOv8Detector, track_with_lk_optical_flow
import os

def process_video(video_path, model_path, output_path=None, detection_interval=10, conf_threshold=0.25):
    """
    Process a video with object detection and optical flow tracking.
    
    Args:
        video_path (str): Path to input video
        model_path (str): Path to YOLOv8 model weights
        output_path (str, optional): Path to save output video
        detection_interval (int): Run detection every N frames
        conf_threshold (float): Confidence threshold for detections
    """
    # Initialize the detector
    detector = YOLOv8Detector(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        # Make sure to use a proper codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' if mp4v doesn't work
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Writing output to {output_path} with dimensions {width}x{height} @ {fps}fps")
    
    # Process variables
    frame_idx = 0
    prev_gray = None
    tracked_boxes = []
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    print(f"Running detection every {detection_interval} frames")
    print(f"Confidence threshold: {conf_threshold}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
        
        # Convert frame to grayscale for optical flow
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Make a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Run detection at intervals
        detection_performed = False
        if frame_idx % detection_interval == 0:
            print(f"Frame {frame_idx}/{total_frames}: Running detection...")
            detections = detector.detect(frame, conf_threshold=conf_threshold)
            tracked_boxes = detections
            detection_performed = True
            print(f"  Found {len(tracked_boxes)} objects")
        else:
            # Use optical flow tracking in between detection frames
            if prev_gray is not None and tracked_boxes:
                tracked_boxes = track_with_lk_optical_flow(prev_gray, curr_gray, tracked_boxes)
        
        # Draw bounding boxes
        boxes_drawn = 0
        for box in tracked_boxes:
            # Make sure box is properly formatted: [x1, y1, x2, y2, conf, cls_id]
            if len(box) < 6:
                print(f"Warning: Invalid box format: {box}")
                continue
                
            x1, y1, x2, y2, conf, cls_id = box
            
            # Convert coordinates to integers
            try:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Validate coordinates
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    print(f"Warning: Invalid box coordinates: {[x1, y1, x2, y2]}")
                    continue
            except:
                print(f"Warning: Could not convert box coordinates to integers: {box}")
                continue
            
            # Class names based on your training
            class_names = ['bag', 'intubator']
            try:
                class_id = int(cls_id)
                label = f"{class_names[class_id]}: {conf:.2f}" if class_id < len(class_names) else f"Class {class_id}: {conf:.2f}"
            except:
                print(f"Warning: Invalid class ID: {cls_id}")
                label = f"Unknown: {conf:.2f}"
            
            # Draw rectangle and label with vibrant colors
            color = (0, 255, 0)  # Green
            if class_id == 0:  # blue_bag
                color = (255, 0, 0)  # Blue (BGR format)
            elif class_id == 1:  # intubator
                color = (0, 0, 255)  # Red (BGR format)
            
            # Draw thicker rectangles for better visibility
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with background for better visibility
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            boxes_drawn += 1
        
        # Display detection status
        status_text = f"Frame: {frame_idx} | "
        if detection_performed:
            status_text += f"Detection: {len(tracked_boxes)} found"
        else:
            status_text += f"Tracking: {len(tracked_boxes)} objects"
            
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Boxes drawn: {boxes_drawn}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Detection & Tracking', display_frame)
        
        # Write frame to output video
        if writer:
            writer.write(display_frame)  # Make sure to write the frame with drawn boxes
        
        # Save current frame for next iteration
        prev_gray = curr_gray.copy()
        frame_idx += 1
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if writer:
        writer.write(display_frame)
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Processing completed. Output saved to: {output_path}")


# Make sure these values are correct when you call the function
if __name__ == "__main__":
    video_name = "2.6.25_sc8"
    VIDEO_PATH = f"./dataset/{video_name}.mp4"
    TRAINED_MODEL_PATH = "./runs/detect/train/weights/best.pt"
    OUTPUT_PATH = f"./detect_{video_name}.mp4"
    
    # Check if input video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Input video not found: {VIDEO_PATH}")
        print(f"Current working directory: {os.getcwd()}")
        print("Available files in dataset directory:")
        if os.path.exists("./dataset"):
            print(os.listdir("./dataset"))
        else:
            print("Dataset directory not found")
        exit(1)
        
    # Check if model exists
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"ERROR: Model file not found: {TRAINED_MODEL_PATH}")
        exit(1)
    
    # Correctly ordered parameters
    process_video(
        video_path=VIDEO_PATH,
        model_path=TRAINED_MODEL_PATH,
        output_path=OUTPUT_PATH,
        detection_interval=10,  # Run detection every 10 frames
        conf_threshold=0.2      # Lower threshold for better recall
    )