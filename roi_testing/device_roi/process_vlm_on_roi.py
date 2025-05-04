import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import os
from models import YOLOv8Detector, track_with_lk_optical_flow
import cv2
import time
import matplotlib.pyplot as plt

# Load the Moondream model
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True
).to(device)

# Function to process ROIs and ask questions
def process_roi_and_predict(image_path, roi_type):
    """
    Process the detected ROI and make predictions using the Moondream VLM.

    Args:
        image_path (str): Path to the image containing the ROI.
        roi_type (str): Type of ROI (e.g., "bag", "intubator").
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load the image
    image = Image.open(image_path)

    # Define questions based on ROI type
    questions = []
    if roi_type == "intubator":
        questions.append("Is a hand over the intubator?")
    elif roi_type == "bag":
        questions.append("Is the bag being squeezed by a hand?")
    else:
        print(f"Unknown ROI type: {roi_type}")
        return

    # Ask questions and get predictions
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = model.query(image, question)["answer"]
        print(f"Answer: {answer}")

def plot_inference_speeds(roi_times, vlm_times, combined_times):
    """
    Plot inference speeds for ROI detection, VLM inference, and combined processing.
    """
    avg_roi = sum(roi_times) / len(roi_times)
    avg_vlm = sum(vlm_times) / len(vlm_times)
    avg_combined = sum(combined_times) / len(combined_times)

    print(f"Average ROI Detection Time: {avg_roi:.4f} seconds")
    print(f"Average VLM Inference Time: {avg_vlm:.4f} seconds")
    print(f"Average Combined Time: {avg_combined:.4f} seconds")

    plt.figure(figsize=(10, 6))
    plt.plot(roi_times, label="ROI Detection", marker="o")
    plt.plot(vlm_times, label="VLM Inference", marker="s")
    plt.plot(combined_times, label="Combined", marker="^")
    plt.xlabel("Frame Index")
    plt.ylabel("Inference Time (seconds)")
    plt.title("Inference Speeds for ROI Detector, VLM, and Combined")
    plt.legend()
    plt.grid()
    plt.show()

def process_video_with_vlm(video_path, model_path, output_path="output_with_vlm.mp4", detection_interval=10, conf_threshold=0.25, vlm_interval=15):
    """
    Process a video, detect ROIs using YOLOv8, and make predictions using Moondream VLM.

    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to YOLOv8 model weights.
        output_path (str): Path to save the output video with annotations.
        detection_interval (int): Run detection every N frames.
        conf_threshold (float): Confidence threshold for detections.
        vlm_interval (int): Run VLM predictions every N frames.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    # Initialize the YOLOv8 detector
    detector = YOLOv8Detector(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    prev_gray = None
    tracked_boxes = []

    roi_times = []
    vlm_times = []
    combined_times = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for optical flow
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Measure ROI detection time
        start_time = time.time()
        if frame_idx % detection_interval == 0:
            detections = detector.detect(frame, conf_threshold=conf_threshold)
            tracked_boxes = detections
        else:
            # Use optical flow tracking in between detection frames
            if prev_gray is not None and tracked_boxes:
                tracked_boxes = track_with_lk_optical_flow(prev_gray, curr_gray, tracked_boxes)
        roi_time = time.time() - start_time
        roi_times.append(roi_time)

        # Measure VLM inference time
        vlm_time = 0
        if frame_idx % vlm_interval == 0:
            for box in tracked_boxes:
                x1, y1, x2, y2, conf, cls_id = box

                # Validate bounding box coordinates
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0] or x2 <= x1 or y2 <= y1:
                    print(f"Warning: Invalid bounding box coordinates: {x1}, {y1}, {x2}, {y2}")
                    continue

                # Crop the frame
                cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]
                if cropped_frame is None or cropped_frame.size == 0:
                    print(f"Warning: Cropped frame is empty for box: {x1}, {y1}, {x2}, {y2}")
                    continue

                # Convert cropped frame to PIL Image
                image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))

                # Define questions based on class ID
                questions = []
                if cls_id == 0:  # Bag
                    questions.append("Is the bag being squeezed by a hand? Answer as 1 for yes, 0 for no.")
                elif cls_id == 1:  # Intubator
                    questions.append("Is the intubator entering the mouth? Answer as 1 for yes, 0 for no.")

                # Ask questions and get predictions
                start_time = time.time()
                for question in questions:
                    answer = model.query(image, question)["answer"]

                    # Draw the ROI and overlay the question and answer
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    text = f"Q: {question} A: {answer}"
                    cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    print(f"Frame {frame_idx}: {text}")
                vlm_time += time.time() - start_time
        vlm_times.append(vlm_time)

        # Measure combined time
        combined_times.append(roi_time + vlm_time)

        # Save current frame for optical flow
        prev_gray = curr_gray.copy()
        frame_idx += 1

        # Write the frame to the output video
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Processing completed. Output saved to {output_path}.")

    # Plot inference speeds
    plot_inference_speeds(roi_times, vlm_times, combined_times)

# Example usage
if __name__ == "__main__":
    video_name = "2.6.25_sc8"
    VIDEO_PATH = f"./dataset/{video_name}.mp4"
    MODEL_PATH = "./runs/detect/train/weights/best.pt"

    process_video_with_vlm(VIDEO_PATH, MODEL_PATH)
    # Replace with actual paths to ROI images
    # process_roi_and_predict("path/to/intubator_roi.jpg", "intubator")
    # process_roi_and_predict("path/to/bag_roi.jpg", "bag")
