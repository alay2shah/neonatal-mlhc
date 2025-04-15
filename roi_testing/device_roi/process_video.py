import cv2
import numpy as np
import argparse
from models import YOLOv8Detector, track_with_lk_optical_flow

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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process variables
    frame_idx = 0
    prev_gray = None
    tracked_boxes = []
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    print(f"Running detection every {detection_interval} frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale for optical flow
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Run detection at intervals
        if frame_idx % detection_interval == 0:
            print(f"Frame {frame_idx}/{total_frames}: Running detection...")
            detections = detector.detect(frame, conf_threshold=conf_threshold)
            tracked_boxes = detections
        else:
            # Use optical flow tracking in between detection frames
            if prev_gray is not None and tracked_boxes:
                tracked_boxes = track_with_lk_optical_flow(prev_gray, curr_gray, tracked_boxes)
        
        # Draw bounding boxes
        for box in tracked_boxes:
            x1, y1, x2, y2, conf, cls_id = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Class names based on your training
            class_names = ['blue_bag', 'intubator']
            class_id = int(cls_id)
            label = f"{class_names[class_id]}: {conf:.2f}" if class_id < len(class_names) else f"Class {class_id}: {conf:.2f}"
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display frame count
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Detection & Tracking', frame)
        
        # Write frame to output video
        if writer:
            writer.write(frame)
        
        # Save current frame for next iteration
        prev_gray = curr_gray.copy()
        frame_idx += 1
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Processing completed. Output saved to: {output_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process video with object detection and optical flow tracking")
    # parser.add_argument("--video", required=True, help="Path to input video file")
    # parser.add_argument("--model", required=True, help="Path to YOLOv8 model weights")
    # parser.add_argument("--output", help="Path to save output video")
    # parser.add_argument("--interval", type=int, default=10, help="Detection interval (frames)")
    # parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    video_name = "2.6.25_sc8"
    VIDEO_PATH = f"./dataset/{video_name}.mp4"
    TRAINED_MODEL_PATH = "./runs/detect/train/weights/best.pt"

    OUTPUT_PATH = f"./detect_{video_name}.mp4"
    
    # args = parser.parse_args()
    
    process_video(
        video_path=VIDEO_PATH,
        model_path=TRAINED_MODEL_PATH,
        output_path=OUTPUT_PATH,
        detection_interval=.25,
        conf_threshold=10
    )