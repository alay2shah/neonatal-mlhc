import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import os
from models import YOLOv8Detector, track_with_lk_optical_flow
import cv2
import time
import matplotlib.pyplot as plt
import moondream as md

# Load the Moondream model
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# model = AutoModelForCausalLM.from_pretrained(
#     "vikhyatk/moondream2",
#     revision="2025-01-09",
#     trust_remote_code=True
# ).to(device)
model_path = "../roi_testing/model/moondream-0_5b-int8.mf.gz"
model = md.vl(model=model_path)
# model = model.to(device)

# Function to process ROIs and ask questions
def process_img_and_predict(image_path, roi_type):
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
        # questions.append("Is the bag being squeezed by a hand?")
        questions.append("Is the bag getting filled with air and squeezed by gloved fingers? Answer as 1 for yes, 0 for no.")
    else:
        print(f"Unknown ROI type: {roi_type}")
        return

    # Ask questions and get predictions
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = model.query(image, question)["answer"]
        print(f"Answer: {answer}")

def process_roi_and_predict(image_path, model_path, roi_type=None, padding = 0.5):
    """
    Process the image, detect ROIs using YOLOv8, crop the ROIs,
    and make predictions using the Moondream VLM.

    Args:
        image_path (str): Path to the image.
        model_path (str): Path to YOLOv8 model weights.
        roi_type (str, optional): Type of ROI to focus on ("bag", "intubator", or None for all).
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Initialize the YOLOv8 detector
    detector = YOLOv8Detector(model_path)
    
    # Detect objects in the image
    detections = detector.detect(img)
    if not detections:
        print("No objects detected in the image.")
        return
    
    # Process each detected object
    for box in detections:
        x1, y1, x2, y2, conf, cls_id = box

        h, w = img.shape[:2]
        roi_width = x2 - x1
        roi_height = y2 - y1
        padding_x = int(roi_width * padding)
        padding_y = int(roi_height * padding)
        
        # Apply padding with boundary checks
        padded_x1 = max(0, x1 - padding_x)
        padded_y1 = max(0, y1 - padding_y)
        padded_x2 = min(w, x2 + padding_x)
        padded_y2 = min(h, y2 + padding_y)
        
        # Filter by roi_type if specified
        if roi_type:
            class_name = "bag" if cls_id == 0 else "intubator"
            if class_name != roi_type:
                continue
        
        # Validate bounding box coordinates
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0] or x2 <= x1 or y2 <= y1:
            print(f"Warning: Invalid bounding box coordinates: {x1}, {y1}, {x2}, {y2}")
            continue
        
        # Crop the frame
        # cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
        cropped_img = img[int(padded_y1):int(padded_y2), int(padded_x1):int(padded_x2)]
        if cropped_img is None or cropped_img.size == 0:
            print(f"Warning: Cropped image is empty for box: {x1}, {y1}, {x2}, {y2}")
            continue
            
        # Save the cropped image (for debugging)
        class_name = "bag" if cls_id == 0 else "intubator"
        output_dir = os.path.join(os.path.dirname(image_path), "roi_crops")
        os.makedirs(output_dir, exist_ok=True)
        crop_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_{class_name}.jpg")
        cv2.imwrite(crop_path, cropped_img)
        
        # Convert cropped frame to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

        # Define questions based on class ID
        questions = []
        if cls_id == 0:  # Bag
            questions.append("Is the bag getting filled with air and squeezed by gloved fingers? Answer as 1 for yes, 0 for no.")
        elif cls_id == 1:  # Intubator
            questions.append("Is the intubator at an angle being inserted into the baby's mouth? Answer as 1 for yes, 0 for no.")
    
        # Ask questions and get predictions
        for question in questions:
            print(f"\nQuestion: {question}")
            answer = model.query(pil_img, question)["answer"]
            print(f"Answer: {answer}")


def plot_inference_speeds(roi_times, vlm_times, combined_times):
    """
    Plot inference speeds for ROI detection, VLM inference, and combined processing,
    including average speeds in frames per second.
    """
    avg_roi = sum(roi_times) / len(roi_times)
    avg_vlm = sum(vlm_times) / len(vlm_times)
    avg_combined = sum(combined_times) / len(combined_times)

    fps_roi = 1 / avg_roi if avg_roi > 0 else 0
    fps_vlm = 1 / avg_vlm if avg_vlm > 0 else 0
    fps_combined = 1 / avg_combined if avg_combined > 0 else 0

    print(f"Average ROI Detection Time: {avg_roi:.4f} seconds ({fps_roi:.2f} FPS)")
    print(f"Average VLM Inference Time: {avg_vlm:.4f} seconds ({fps_vlm:.2f} FPS)")
    print(f"Average Combined Time: {avg_combined:.4f} seconds ({fps_combined:.2f} FPS)")

    plt.figure(figsize=(10, 6))
    plt.plot(roi_times, label="ROI Detection", marker="o")
    plt.plot(vlm_times, label="VLM Inference", marker="s")
    plt.plot(combined_times, label="Combined", marker="^")
    plt.axhline(y=avg_roi, color='blue', linestyle='--', label=f"Avg ROI: {avg_roi:.4f}s ({fps_roi:.2f} FPS)")
    plt.axhline(y=avg_vlm, color='orange', linestyle='--', label=f"Avg VLM: {avg_vlm:.4f}s ({fps_vlm:.2f} FPS)")
    plt.axhline(y=avg_combined, color='green', linestyle='--', label=f"Avg Combined: {avg_combined:.4f}s ({fps_combined:.2f} FPS)")
    plt.xlabel("Frame Index")
    plt.ylabel("Inference Time (seconds)")
    plt.title("Inference Speeds for ROI Detector, VLM, and Combined")
    plt.legend()
    plt.grid()
    plt.show()

def process_video_with_vlm_and_roi(video_path, model_path, output_path="output_with_vlm.mp4", detection_interval=15, conf_threshold=0.25, vlm_interval=30, padding = 0.5):
    """
    Process a video, detect ROIs using YOLOv8, and make predictions using Moondream VLM.
    Keeps inference results visible at all times at the bottom of the video.

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

    # Dictionary to store the last inference result for each ROI class
    last_inference = {0: "No inference yet", 1: "No inference yet"}  # Class ID -> Result
    last_answers = {0: "N/A", 1: "N/A"}  # Class ID -> Answer

    # Area for text at the bottom
    text_area_height = 100
    bottom_padding = 10

    roi_times = []
    vlm_times = []
    combined_times = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create a larger frame with space for text at the bottom
        display_frame = frame.copy()

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
                
                h, w = frame.shape[:2]
                roi_width = x2 - x1
                roi_height = y2 - y1
                padding_x = int(roi_width * padding)
                padding_y = int(roi_height * padding)
                
                # Apply padding with boundary checks
                padded_x1 = max(0, x1 - padding_x)
                padded_y1 = max(0, y1 - padding_y)
                padded_x2 = min(w, x2 + padding_x)
                padded_y2 = min(h, y2 + padding_y)

                # Validate bounding box coordinates
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0] or x2 <= x1 or y2 <= y1:
                    print(f"Warning: Invalid bounding box coordinates: {x1}, {y1}, {x2}, {y2}")
                    continue

                # Crop the frame
                cropped_frame = frame[int(padded_y1):int(padded_y2), int(padded_x1):int(padded_x2)]
                if cropped_frame is None or cropped_frame.size == 0:
                    print(f"Warning: Cropped frame is empty for box: {x1}, {y1}, {x2}, {y2}")
                    continue

                # Convert cropped frame to PIL Image
                image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))

                # Define questions based on class ID
                questions = []
                if cls_id == 0:  # Bag
                    questions.append("Is the bag getting filled with air and squeezed by gloved fingers? Answer as 1 for yes, 0 for no.")
                elif cls_id == 1:  # Intubator
                    questions.append("Is the intubator at an angle being inserted into the baby's mouth? Answer as 1 for yes, 0 for no.")

                # Ask questions and get predictions
                start_time = time.time()
                for question in questions:
                    answer = model.query(image, question)["answer"]
                    
                    # Update the last inference result for this class
                    last_inference[cls_id] = question
                    last_answers[cls_id] = answer
                    
                    print(f"Frame {frame_idx}: Q: {question} A: {answer}")
                vlm_time += time.time() - start_time
        vlm_times.append(vlm_time)

        # Measure combined time
        combined_times.append(roi_time + vlm_time)

        # Draw ROIs for all tracked boxes
        for box in tracked_boxes:
            x1, y1, x2, y2, conf, cls_id = box
            
            # Validate coordinates
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0] or x2 <= x1 or y2 <= y1:
                continue
                
            # Draw the ROI box
            color = (0, 255, 0)  # Green by default
            if cls_id == 0:  # Bag
                color = (255, 0, 0)  # Blue (BGR)
            elif cls_id == 1:  # Intubator
                color = (0, 0, 255)  # Red (BGR)
                
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Label the ROI
            class_name = "Bag" if cls_id == 0 else "Intubator"
            cv2.putText(display_frame, f"{class_name}: {conf:.2f}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw inference results at the bottom of the frame
        # Create a dark semi-transparent overlay for text at the bottom
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, height - text_area_height), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        # Add inference results for each class
        y_position = height - text_area_height + bottom_padding + 20
        for cls_id in last_inference:
            class_name = "Bag" if cls_id == 0 else "Intubator"
            result_text = f"{class_name}: Q: {last_inference[cls_id]}"
            answer_text = f"A: {last_answers[cls_id]}"
            
            # Draw class name and question
            cv2.putText(display_frame, result_text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw answer on next line
            cv2.putText(display_frame, answer_text, (10, y_position + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_position += 60  # Move down for the next class

        # Save current frame for optical flow
        prev_gray = curr_gray.copy()
        frame_idx += 1

        # Write the frame to the output video
        writer.write(display_frame)
        
        # Display the frame if needed
        cv2.imshow('Video with VLM', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Processing completed. Output saved to {output_path}.")

    # Plot inference speeds
    plot_inference_speeds(roi_times, vlm_times, combined_times)

def process_video_with_vlm_only(video_path, output_path="output_vlm_only.mp4", vlm_interval=30):
    """
    Process a video with Moondream VLM only (no ROI detection),
    asking the same question every N frames and showing results.
    
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output video with annotations.
        vlm_interval (int): Run VLM predictions every N frames.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # The question to ask for every frame
    question = "Is the bag getting filled with air and squeezed by gloved fingers? Answer as 1 for yes, 0 for no."
    
    # Track the last answer
    last_answer = "No inference yet"
    
    # Area for text at the bottom
    text_area_height = 100
    bottom_padding = 10

    # Performance tracking
    vlm_times = []
    
    # Process frames
    frame_idx = 0
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Running VLM every {vlm_interval} frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Run VLM at specified intervals
        if frame_idx % vlm_interval == 0:
            print(f"\nProcessing frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
            
            # Convert frame to PIL image for VLM
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Measure VLM inference time
            start_time = time.time()
            
            # Run VLM query
            answer = model.query(pil_image, question)["answer"]
            last_answer = answer
            
            vlm_time = time.time() - start_time
            vlm_times.append(vlm_time)
            
            print(f"Frame {frame_idx}: Q: {question} A: {answer} ({vlm_time:.2f}s)")
        
        # Create a dark semi-transparent overlay for text at the bottom
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, height - text_area_height), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        # Add inference results at the bottom
        y_position = height - text_area_height + bottom_padding + 20
        result_text = f"Q: {question}"
        answer_text = f"A: {last_answer}"
        
        # Draw question and answer
        cv2.putText(display_frame, result_text, (10, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, answer_text, (10, y_position + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add frame number and processing info
        status_text = f"Frame: {frame_idx}/{total_frames}"
        if frame_idx % vlm_interval == 0:
            status_text += f" (VLM: {vlm_time:.2f}s)"
        
        cv2.putText(display_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Write frame to video
        writer.write(display_frame)
        
        # Display the frame
        cv2.imshow('Video with VLM Only', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    # Calculate average inference time
    if vlm_times:
        avg_vlm_time = sum(vlm_times) / len(vlm_times)
        print(f"Processing completed. Output saved to {output_path}.")
        print(f"Average VLM inference time: {avg_vlm_time:.4f} seconds")
        
        # Plot inference times
        plt.figure(figsize=(10, 6))
        plt.plot(vlm_times, label="VLM Inference Time", marker="o")
        plt.axhline(y=avg_vlm_time, color='r', linestyle='-', label=f"Average: {avg_vlm_time:.4f}s")
        plt.xlabel("Frame Index")
        plt.ylabel("Inference Time (seconds)")
        plt.title("VLM Inference Times")
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("No frames were processed with VLM.")

# Example usage
if __name__ == "__main__":
    video_name = "2.6.25_sc8"
    VIDEO_PATH = f"./dataset/{video_name}.mp4"
    MODEL_PATH = "./runs/detect/train/weights/best.pt"

    # process_video_with_vlm_and_roi(VIDEO_PATH, MODEL_PATH)
    # Replace with actual paths to ROI images
    # process_roi_and_predict("path/to/intubator_roi.jpg", "intubator")
    # test_image_path = "./dataset/testing_data/ppv_action.png"
    # print("With ROI:")
    # process_roi_and_predict(test_image_path, MODEL_PATH, "bag")  # Fix argument order
    # print("Without ROI:")
    # process_img_and_predict(test_image_path, "bag")

    # Process with ROI detection
    roi_output_path = f"./output_with_roi_{video_name}_0_5_b.mp4"
    process_video_with_vlm_and_roi(VIDEO_PATH, MODEL_PATH, output_path=roi_output_path)
    
    # Process with VLM only
    vlm_output_path = f"./output_vlm_only_{video_name}_0_5_b.mp4"
    process_video_with_vlm_only(VIDEO_PATH, output_path=vlm_output_path)

