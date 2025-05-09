import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import os
from models import YOLOv8Detector, track_with_lk_optical_flow
import cv2
import time
import matplotlib.pyplot as plt
import moondream as md
from plot_vlm_inference import  plot_binary_predictions_over_time
import csv
import numpy as np

# Load the Moondream model
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True
).to(device)
# model_path = "../roi_testing/model/moondream-0_5b-int8.mf.gz"


# model = md.vl(model=model_path)


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


def plot_inference_speeds(roi_times, vlm_times, combined_times, save_dir="./figures"):
    """
    Plot inference speeds for ROI detection, VLM inference, and combined processing,
    including average speeds in frames per second.
    
    Args:
        roi_times (list): List of ROI detection times in seconds.
        vlm_times (list): List of VLM inference times in seconds.
        combined_times (list): List of combined processing times in seconds.
        save_dir (str): Directory to save the figure.
    """
    # Create the figures directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    avg_roi = sum(roi_times) / len(roi_times)
    avg_vlm = sum(vlm_times) / len(vlm_times)
    avg_combined = sum(combined_times) / len(combined_times)

    fps_roi = 1 / avg_roi if avg_roi > 0 else 0
    fps_vlm = 1 / avg_vlm if avg_vlm > 0 else 0
    fps_combined = 1 / avg_combined if avg_combined > 0 else 0

    print(f"Average ROI Detection Time: {avg_roi:.4f} seconds ({fps_roi:.2f} FPS)")
    print(f"Average VLM Inference Time: {avg_vlm:.4f} seconds ({fps_vlm:.2f} FPS)")
    print(f"Average Combined Time: {avg_combined:.4f} seconds ({fps_combined:.2f} FPS)")
    
    # Save the data to a CSV file
    csv_filename = os.path.join(save_dir, "inference_speeds_comparison.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame Index', 'ROI Detection Time (s)', 'VLM Inference Time (s)', 'Combined Time (s)'])
        for i, (r, v, c) in enumerate(zip(roi_times, vlm_times, combined_times)):
            writer.writerow([i, r, v, c])
    print(f"Data saved to {csv_filename}")

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
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(save_dir, "inference_speeds_comparison.png")
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Figure saved to {filename}")

def process_video_with_vlm_and_roi(
    video_path, model_path, output_path="output_with_vlm.mp4", detection_interval=15, conf_threshold=0.25,
    vlm_interval=30, padding=0.5, multi_frame=True, frames_to_capture=15, frame_interval=3, use_grid=False, plots=True
):
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
        padding (float): Padding factor for ROI bounding boxes.
        multi_frame (bool): Whether to use multiple frames for VLM prediction.
        frames_to_capture (int): Number of frames to capture in a sequence.
        frame_interval (int): Interval between frames in a sequence.
        use_grid (bool): If True, use a grid of 4 images for inference.
        plots (bool): If True, plot inference speeds and binary predictions.
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    
    # Track predictions and timestamps by class for plotting
    prediction_timestamps = {0: [], 1: []}  # Class ID -> list of timestamps
    prediction_answers = {0: [], 1: []}     # Class ID -> list of answers
    prediction_questions = {0: "", 1: ""}   # Class ID -> question
    prediction_frame_indices = {0: [], 1: []}  # Class ID -> list of frame indices (make sure this line exists)
    cumulative_time = 0
    
    # For multi-frame processing
    frame_buffer = {}  # cls_id -> list of (frame_idx, PIL image)
    all_frames_buffer = {}  # Store more frames for grid processing
    prediction_frame_indices = {0: [], 1: []}
    
    # If using grid, make sure to collect enough frames
    grid_frames_needed = 4
    frames_to_keep = max(vlm_interval, grid_frames_needed * 15) if use_grid else vlm_interval

    # Track frames since last VLM inference for each class
    last_vlm_frame = 0
    
    # For multi-frame processing
    frame_buffer = {}  # cls_id -> list of (frame_idx, PIL image)
    all_frames_buffer = {}  # Store more frames for grid processing - will be reset at each VLM interval
    prediction_frame_indices = {0: [], 1: []}


    # Keep track of the starting frame for the current VLM interval
    vlm_interval_start = 0
    
    # Create buffers that are specific to the current VLM interval
    current_interval_buffers = {0: [], 1: []}
    
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start of new VLM interval - reset buffers
        if frame_idx % vlm_interval == 0:
            vlm_interval_start = frame_idx
            current_interval_buffers = {0: [], 1: []}
            print(f"Starting new VLM interval at frame {frame_idx}")

        # Create a larger frame with space for text at the bottom
        display_frame = frame.copy()

        # Convert frame to grayscale for optical flow
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reset all_frames_buffer at each VLM interval
        if frame_idx % vlm_interval == 0:
            all_frames_buffer = {0: [], 1: []}
            last_vlm_frame = frame_idx

        # Measure ROI detection time
        start_time = time.time()
        if frame_idx % detection_interval == 0:
            detections = detector.detect(frame, conf_threshold=conf_threshold)
            tracked_boxes = detections
            # Reset frame buffer when we get new detections
            # frame_buffer = {0: [], 1: []}
        else:
            # Use optical flow tracking in between detection frames
            if prev_gray is not None and tracked_boxes:
                tracked_boxes = track_with_lk_optical_flow(prev_gray, curr_gray, tracked_boxes)
        roi_time = time.time() - start_time
        roi_times.append(roi_time)

        # Init per-class buffers if needed
        for cls_id in [0, 1]:
            if cls_id not in all_frames_buffer:
                all_frames_buffer[cls_id] = []
        
        # Collect frames for the buffer if we have detections
        for box in tracked_boxes:
            x1, y1, x2, y2, conf, cls_id = box
            cls_id = int(cls_id)  # Ensure it's an integer for indexing
            
            # Skip invalid boxes
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0] or x2 <= x1 or y2 <= y1:
                continue
                
            # Apply padding to bounding box
            h, w = frame.shape[:2]
            roi_width = x2 - x1
            roi_height = y2 - y1
            padding_x = int(roi_width * padding)
            padding_y = int(roi_height * padding)
            
            padded_x1 = max(0, x1 - padding_x)
            padded_y1 = max(0, y1 - padding_y)
            padded_x2 = min(w, x2 + padding_x)
            padded_y2 = min(h, y2 + padding_y)
            
            # Crop the frame
            cropped_frame = frame[int(padded_y1):int(padded_y2), int(padded_x1):int(padded_x2)]
            if cropped_frame is None or cropped_frame.size == 0:
                continue
                
            # Convert to PIL for VLM
            pil_img = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            
            # Initialize buffer for this class if needed
            if cls_id not in frame_buffer:
                frame_buffer[cls_id] = []                
            frame_buffer[cls_id].append((frame_idx, pil_img))
            if len(frame_buffer[cls_id]) > frames_to_capture * frame_interval:
                frame_buffer[cls_id].pop(0)
                
            # Also add to all_frames_buffer for grid
            all_frames_buffer[cls_id].append((frame_idx, pil_img))
            if len(all_frames_buffer[cls_id]) > frames_to_keep:
                all_frames_buffer[cls_id].pop(0)

            # Only add frames that are part of the current VLM interval
            if frame_idx >= vlm_interval_start:
                cls_id = int(cls_id)
                if cls_id not in current_interval_buffers:
                    current_interval_buffers[cls_id] = []
                
                # Store the current ROI frame
                current_interval_buffers[cls_id].append((frame_idx, pil_img))
                
                # Debug visualization to show buffer growth
                class_name = "Bag" if cls_id == 0 else "Intubator"
                cv2.putText(display_frame, f"{class_name} buffer: {len(current_interval_buffers[cls_id])} frames", 
                           (10, 30 + cls_id * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # # Measure VLM inference time
        # vlm_time = 0
        # if frame_idx % vlm_interval == 0:
        #     print(f"Running VLM inference at frame {frame_idx}...")
            
        #     # Debug print all buffer sizes
        #     for cls_id in [0, 1]:
        #         buffer_size = len(all_frames_buffer.get(cls_id, []))
        #         print(f"Frame {frame_idx}: Class {cls_id} has {buffer_size} frames in buffer")
            
        #     for cls_id in list(all_frames_buffer.keys()):
        #         answer_to_store = None
                
        #         # Skip if no ROIs were detected for this class
        #         if not all_frames_buffer.get(cls_id, []):
        #             print(f"No {('Bag' if cls_id == 0 else 'Intubator')} detected in this interval, skipping...")
        #             continue
                
        #         # --- Use grid if requested ---
        #         if use_grid:
        #             buffer_items = all_frames_buffer.get(cls_id, [])
        #             if len(buffer_items) >= 1:  # At least one frame is needed
        #                 # Determine how many frames to use (up to 4)
        #                 grid_frames_needed = min(4, len(buffer_items))
                        
        #                 # Always select frames evenly spaced throughout the buffer
        #                 if len(buffer_items) >= grid_frames_needed:
        #                     # Use numpy linspace to get evenly spaced indices
        #                     indices = np.linspace(0, len(buffer_items)-1, grid_frames_needed, dtype=int)
        #                     selected_items = [buffer_items[i] for i in indices]
                            
        #                     # Debug print the selected indices
        #                     frame_nums = [item[0] for item in selected_items]
        #                     print(f"Selected {grid_frames_needed} frames at indices: {frame_nums} from buffer of {len(buffer_items)}")
                            
        #                     # Extract just the PIL images
        #                     images_for_grid = [item[1] for item in selected_items]
                            
        #                     # Create question based on class
        #                     if cls_id == 0:  # Bag
        #                         # question = "Is the blue bag being squeezed by hands and filled with air? Answer as 1 for yes, 0 for no."
        #                         question = "Is a gloved hand grabbing the bag and either squeezing it or filling it with air? Answer as 1 for yes, 0 for no."
        #                         prediction_questions[0] = question
        #                     else:  # Intubator
        #                         question = "Is the intubator at an angle being inserted into the baby's mouth? Answer as 1 for yes, 0 for no."
        #                         prediction_questions[1] = question
                            
        #                     # Create and save the grid image
        #                     try:
        #                         grid_img = create_image_grid(images_for_grid)
        #                         grid_path = os.path.join(debug_dir, f"grid_cls{cls_id}_frame{frame_idx}.jpg")
        #                         grid_img.save(grid_path)
        #                         print(f"Saved grid with {len(images_for_grid)} images to {grid_path}")
                                
        #                         # Process with VLM
        #                         start_time = time.time()
        #                         answer_to_store = process_image_grid(model, images_for_grid, question)
        #                         vlm_time += time.time() - start_time
                                
        #                         # Update results
        #                         last_inference[cls_id] = question
        #                         last_answers[cls_id] = answer_to_store
        #                         print(f"Frame {frame_idx}: Grid Q: {question} A: {answer_to_store}")
        #                     except Exception as e:
        #                         print(f"Error creating/processing grid: {e}")
        #                         import traceback
        #                         traceback.print_exc()
                
        #         # Only store prediction if we made one
        #         if answer_to_store is not None:
        #             prediction_frame_indices[cls_id].append(frame_idx)
        #             prediction_answers[cls_id].append(answer_to_store)
                    
        #     # Clear frame buffers after processing
        #     all_frames_buffer = {0: [], 1: []}

        # Run VLM at the end of each interval
        vlm_time = 0
        if frame_idx > 0 and (frame_idx + 1) % vlm_interval == 0:
            print(f"Running VLM inference at frame {frame_idx} (end of interval)")
            
            # Debug print buffer contents
            for cls_id in [0, 1]:
                class_name = "Bag" if cls_id == 0 else "Intubator"
                buffer_len = len(current_interval_buffers.get(cls_id, []))
                print(f"Frame {frame_idx}: {class_name} has {buffer_len} frames in current interval buffer")
                
                # Process each class if we have frames
                if buffer_len > 0:
                    buffer_items = current_interval_buffers[cls_id]
                    
                    if use_grid:
                        # We want to select 4 frames evenly distributed across the interval
                        grid_size = min(4, buffer_len)
                        
                        if buffer_len >= grid_size:
                            # Select frames evenly distributed through the buffer
                            indices = np.linspace(0, buffer_len-1, grid_size, dtype=int)
                            selected_items = [buffer_items[i] for i in indices]
                            selected_frames = [item[1] for item in selected_items]
                            frame_indices = [item[0] for item in selected_items]
                            
                            print(f"Selected {grid_size} frames at indices {frame_indices} from buffer of {buffer_len} frames")
                            
                            # Save individual frames for debugging
                            # for i, img in enumerate(selected_frames):
                            #     img_debug_dir = os.path.join(debug_dir, f"interval_{frame_idx//vlm_interval}_cls{cls_id}")
                            #     os.makedirs(img_debug_dir, exist_ok=True)
                            #     img.save(os.path.join(img_debug_dir, f"frame_{i}_idx_{frame_indices[i]}.jpg"))
                            
                            # Define question based on class
                            if cls_id == 0:  # Bag
                                # question = "Is the blue bag being squeezed by hands and filled with air? Answer as 1 for yes, 0 for no."
                                question = "Is a gloved hand squeezing the bag tightly? Answer as 1 for yes, 0 for no."
                                prediction_questions[0] = question
                            else:  # Intubator
                                question = "Is the intubator at an angle being inserted into the baby's mouth? Answer as 1 for yes, 0 for no."
                                prediction_questions[1] = question
                            
                            # Create and process grid
                            try:
                                grid_img = create_image_grid(selected_frames)
                                # grid_path = os.path.join(debug_dir, f"grid_cls{cls_id}_interval_{frame_idx//vlm_interval}.jpg")
                                # grid_img.save(grid_path)
                                
                                start_time = time.time()
                                answer = process_image_grid(model, selected_frames, question)
                                vlm_time += time.time() - start_time
                                
                                last_inference[cls_id] = question
                                last_answers[cls_id] = answer
                                
                                # Save prediction for plotting
                                prediction_frame_indices[cls_id].append(frame_idx)
                                prediction_answers[cls_id].append(answer)
                                
                                print(f"Frame {frame_idx}: {class_name} grid analysis: {answer}")
                            except Exception as e:
                                print(f"Error processing grid: {e}")
                                import traceback
                                traceback.print_exc()
                    else:
                        # When not using grid, directly use the latest ROI
                        try:
                            # Select the most recent frame from the buffer
                            latest_frame_idx, latest_roi_img = buffer_items[-1]
                            
                            # Define question based on class
                            if cls_id == 0:  # Bag
                                question = "Is the blue bag getting filled with air or being squeezed by gloved fingers? Answer as 1 for yes, 0 for no."
                                prediction_questions[0] = question
                            else:  # Intubator
                                question = "Is the intubator at an angle being inserted into the baby's mouth? Answer as 1 for yes, 0 for no."
                                prediction_questions[1] = question
                            
                            # Process with VLM directly (single image)
                            start_time = time.time()
                            answer = model.query(latest_roi_img, question)["answer"]
                            vlm_time += time.time() - start_time
                            
                            # Update results
                            last_inference[cls_id] = question
                            last_answers[cls_id] = answer
                            
                            # Save prediction for plotting
                            prediction_frame_indices[cls_id].append(frame_idx)
                            prediction_answers[cls_id].append(answer)
                            
                            print(f"Frame {frame_idx}: {class_name} direct analysis: Q: {question} A: {answer}")
                            
                   
                        except Exception as e:
                            print(f"Error in direct ROI processing: {e}")
                            import traceback
                            traceback.print_exc()
        
        vlm_times.append(vlm_time)
        cumulative_time += vlm_time + roi_time
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

        # Display information about multi-frame processing
        if multi_frame:
            for cls_id in frame_buffer:
                if frame_buffer[cls_id]:
                    class_name = "Bag" if cls_id == 0 else "Intubator"
                    buffer_info = f"{class_name} frames: {len(frame_buffer[cls_id])}/{frames_to_capture * frame_interval}"
                    cv2.putText(display_frame, buffer_info, (10, 60 + cls_id * 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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

    if plots == True:
        # Plot inference speeds
        plot_inference_speeds(roi_times, vlm_times, combined_times)

        # Create a base filename for figures based on the video name
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        figures_dir = f"./figures/{video_name}/roi"
        os.makedirs(figures_dir, exist_ok=True)
            # Plot predictions for each class
        for cls_id in [0, 1]:
            class_name = "Bag" if cls_id == 0 else "Intubator"
            if prediction_frame_indices[cls_id]:  # Only plot if we have data
                try:
                    # Use the correct plotting function with proper parameters
                    plot_binary_predictions_over_time(
                        prediction_frame_indices[cls_id],  # Use frame indices directly
                        prediction_answers[cls_id],
                        "VLM and ROI", 
                        prediction_questions[cls_id],
                        class_name,
                        save_dir=figures_dir,
                        fps=fps,
                        use_frame_indices=True  # Important: Tell the function to convert indices to time
                    )
                except Exception as e:
                    print(f"Error plotting binary predictions for {class_name}: {e}")
                    import traceback
                    traceback.print_exc()
    

    
    # Add debug printing to see what's in the prediction data
    for cls_id in [0, 1]:
        class_name = "Bag" if cls_id == 0 else "Intubator"
        print(f"\n{class_name} predictions:")
        print(f"  Frame indices: {prediction_frame_indices[cls_id]}")
        print(f"  Answers: {prediction_answers[cls_id]}")
        print(f"  Question: {prediction_questions[cls_id]}")
    
   



def create_image_grid(images, max_images=4):
    """
    Create a grid of images to process as a single image.
    
    Args:
        images: List of PIL Image objects
        max_images: Maximum number of images to include in grid
        
    Returns:
        PIL Image containing a grid of the input images
    """
    images = images[:max_images]
    n = len(images)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    width = min(img.width for img in images)
    height = min(img.height for img in images)
    images = [img.resize((width, height)) for img in images]
    grid_width = width * cols
    grid_height = height * rows
    grid = Image.new('RGB', (grid_width, grid_height))
    for i, img in enumerate(images):
        grid.paste(img, (width * (i % cols), height * (i // cols)))
    return grid

def process_image_grid(model, images, question):
    """
    Process multiple images by creating a grid and asking about it.
    
    Args:
        model: Moondream VLM model
        images: List of PIL Image objects
        question: Question to ask about the image sequence
        
    Returns:
        Analysis of the image grid
    """
    grid = create_image_grid(images)
    grid_prompt = f"This image shows a grid of {len(images)} sequential frames from a video. {question}"
    return model.query(grid, grid_prompt)["answer"]

# Example usage in your pipeline:
# Suppose you have a list of PIL images (e.g., sampled every 15 frames)
# images = [img1, img2, img3, img4]
# question = "Describe how the bag is being squeezed over time. Is there a change in squeezing?"
# answer = process_image_grid(model, images, question)
# print(answer)

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
    roi_output_path = f"./output_with_roi_{video_name}_2b_grid.mp4"
    process_video_with_vlm_and_roi(VIDEO_PATH, MODEL_PATH, output_path=roi_output_path, multi_frame = False, use_grid = False)
    
    # Process with VLM only
    # vlm_output_path = f"./output_vlm_only_{video_name}_0_5_b.mp4"
    # process_video_with_vlm_only(VIDEO_PATH, output_path=vlm_output_path)































































































































