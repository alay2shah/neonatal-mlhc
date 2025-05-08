import time
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from PIL import Image
import cv2
from models import YOLOv8Detector

# Initialize models
detector = YOLOv8Detector("./runs/detect/train/weights/best.pt")
vlm_model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True
).to("cpu")  # Use CPU for consistent benchmarking

# Measure inference speeds
def measure_inference_speeds(video_path, num_frames=50):
    cap = cv2.VideoCapture(video_path)
    roi_times = []
    vlm_times = []
    combined_times = []

    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Measure ROI detection time
        start_time = time.time()
        detections = detector.detect(frame, conf_threshold=0.25)
        roi_time = time.time() - start_time
        roi_times.append(roi_time)

        # Measure VLM inference time for each ROI
        vlm_time = 0
        for box in detections:
            x1, y1, x2, y2, _, _ = box
            cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]
            image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))

            start_time = time.time()
            _ = vlm_model.query(image, "Is there a hand in the ROI?")
            vlm_time += time.time() - start_time
        vlm_times.append(vlm_time)

        # Measure combined time
        combined_times.append(roi_time + vlm_time)

        frame_idx += 1

    cap.release()
    return roi_times, vlm_times, combined_times

# Plot inference speeds
def plot_inference_speeds(roi_times, vlm_times, combined_times):
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

# Example usage
if __name__ == "__main__":
    video_path = "./dataset/2.6.25_sc8.mp4"
    roi_times, vlm_times, combined_times = measure_inference_speeds(video_path)
    plot_inference_speeds(roi_times, vlm_times, combined_times)
