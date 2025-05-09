import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def plot_binary_predictions_over_time(timestamps, predictions, approach_name, question, class_name=None, save_dir="./figures", fps=30, frame_interval=30, use_frame_indices=False):
    """
    Plot binary predictions (yes=1, no=0) over time with tick marks at 30-second intervals.
    
    Args:
        timestamps (list): List of timestamps (either frame indices or seconds).
        predictions (list): List of prediction answers as strings.
        approach_name (str): Name of the approach (e.g., "VLM Only" or "VLM and ROI").
        question (str): The question that was asked.
        class_name (str, optional): Class name if using ROI approach (e.g., "Bag" or "Intubator").
        save_dir (str): Directory to save the figure and CSV file.
        fps (float): Frames per second of the video.
        frame_interval (int): Number of frames between predictions.
        use_frame_indices (bool): If True, treat timestamps as frame indices.
    """
    # Create the figures directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if we have any data
    if not timestamps or not predictions:
        print(f"No data to plot for {approach_name}" + (f" - {class_name}" if class_name else ""))
        return
    
    if len(timestamps) != len(predictions):
        print(f"Error: Timestamps ({len(timestamps)}) and predictions ({len(predictions)}) must have the same length")
        return
    
    # Print debug info
    print(f"Debug - First 5 raw timestamps: {timestamps[:5]}")
        
    # Convert all predictions to binary (1=yes, 0=no)
    binary_predictions = []
    
    # Comprehensive list of terms for classification
    yes_terms = ["yes", "1", "true", "y", "yep", "yeah", "correct", "positive", "majority vote"]
    no_terms = ["no", "0", "false", "n", "nope", "negative", "nay"]
    
    for pred in predictions:
        pred_lower = str(pred).strip().lower()
        
        # Check for YES variants (including if any yes_term is found in the prediction)
        if any(yes_term in pred_lower for yes_term in yes_terms):
            binary_predictions.append(1)
            print(f"Classified as YES: {pred}")
        # Check for NO variants (including if any no_term is found in the prediction)
        elif any(no_term in pred_lower for no_term in no_terms):
            binary_predictions.append(0)
            print(f"Classified as NO: {pred}")
        else:
            # If we can't determine, use NaN and print a warning
            binary_predictions.append(np.nan)
            print(f"WARNING: Could not classify prediction: '{pred}' - treating as NaN")
    
    # Compute video times from frame indices if requested
    frame_indices = timestamps.copy()
    if use_frame_indices:
        # timestamps are frame indices (e.g., [0, 30, 60, ...])
        video_times = [int(idx) / fps for idx in timestamps]
        print(f"Debug - Converting frame indices to seconds using fps={fps}")
        print(f"Debug - First 5 video times: {video_times[:5]}")
    else:
        # timestamps are already in seconds or another time format
        video_times = timestamps
        # Try to detect if these might actually be frame indices
        if timestamps and max(timestamps) > 1000:  # If timestamps are large, they might be frame indices
            print(f"Warning: Timestamps appear to be large values ({max(timestamps)}). Consider using use_frame_indices=True.")

    # Save the binary predictions to a CSV file with frame indices
    safe_approach = approach_name.replace(' ', '_').lower()
    safe_class = f"_{class_name.lower()}" if class_name else ""
    csv_filename = os.path.join(save_dir, f"binary_predictions_{safe_approach}{safe_class}.csv")
    
    # Write to CSV with both timestamps and frame indices
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame Index', 'Video Time (s)', 'Video Time (MM:SS)', 'Original Prediction', 'Binary Prediction (1=Yes, 0=No)'])
        for i, (f_idx, t, orig, binary) in enumerate(zip(frame_indices, video_times, predictions, binary_predictions)):
            mm_ss = f"{int(t // 60):02d}:{int(t % 60):02d}"
            writer.writerow([f_idx, t, mm_ss, orig, binary if not np.isnan(binary) else "Unknown"])
    print(f"Binary predictions saved to {csv_filename}")
    
    # Calculate tick marks at 30-second intervals for x-axis
    max_time = max(video_times) if video_times else 0
    tick_interval = 30  # seconds
    tick_times = list(range(0, int(max_time) + tick_interval, tick_interval))
    tick_labels = [f"{t // 60}:{t % 60:02d}" for t in tick_times]

    # Plot the binary predictions
    plt.figure(figsize=(12, 6))
    
    # Create a scatter plot with colored points
    valid_indices = [i for i, val in enumerate(binary_predictions) if not np.isnan(val)]
    valid_times = [video_times[i] for i in valid_indices]
    valid_predictions = [binary_predictions[i] for i in valid_indices]
    
    if valid_times:  # Only plot if we have valid data
        # Plot the line connecting all points
        plt.plot(valid_times, valid_predictions, 'b-', alpha=0.5)
        
        # Plot points for "yes" in green
        yes_indices = [i for i, val in enumerate(valid_predictions) if val == 1]
        if yes_indices:
            yes_times = [valid_times[i] for i in yes_indices]
            yes_predictions = [valid_predictions[i] for i in yes_indices]
            plt.scatter(yes_times, yes_predictions, color='green', s=80, label="Yes", marker="o")
        
        # Plot points for "no" in red
        no_indices = [i for i, val in enumerate(valid_predictions) if val == 0]
        if no_indices:
            no_times = [valid_times[i] for i in no_indices]
            no_predictions = [valid_predictions[i] for i in no_indices]
            plt.scatter(no_times, no_predictions, color='red', s=80, label="No", marker="x")

    # Add summary of binary classification
    yes_count = sum(1 for val in binary_predictions if val == 1)
    no_count = sum(1 for val in binary_predictions if val == 0)
    unknown_count = sum(1 for val in binary_predictions if np.isnan(val))
    summary_text = f"Yes: {yes_count}, No: {no_count}"
    if unknown_count > 0:
        summary_text += f", Unknown: {unknown_count}"
    plt.figtext(0.5, 0.95, summary_text, ha='center', fontsize=12)

    # Set y-axis to show only 0 and 1
    plt.yticks([0, 1], ["No (0)", "Yes (1)"])
    plt.ylim(-0.1, 1.1)  # Add some padding

    # Set x-axis tick marks at 30-second intervals
    plt.xlabel("Video Time (minutes:seconds)")
    plt.xticks(tick_times, tick_labels, rotation=45)
    
    # Add title and grid
    title = f"Binary Predictions Over Time ({approach_name})"
    if class_name:
        title += f" - {class_name}"
    plt.title(title)
    
    # Add question as subtitle
    plt.figtext(0.5, 0.01, f"Question: {question}", ha='center', fontsize=10)
    
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    figure_filename = os.path.join(save_dir, f"binary_predictions_{safe_approach}{safe_class}.png")
    plt.savefig(figure_filename, dpi=300)
    plt.show()
    print(f"Binary predictions plot saved to {figure_filename}")

# Example usage after processing videos
# Assuming `vlm_times_only` and `vlm_times_roi` are lists of VLM inference times for the two approaches
# plot_vlm_inference_over_time(vlm_times_only, "VLM Only")
# plot_vlm_inference_over_time(vlm_times_roi, "VLM and ROI")
# plot_binary_predictions_over_time(timestamps, predictions, "VLM and ROI", "Is the bag being squeezed?", class_name="Bag")
if __name__ == "__main__":
    # Example data for testing - import from CSV file
    csv_path = "./figures/2.6.25_sc8/roi/vlm_predictions_vlm_and_roi_bag.csv"
    
    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
    else:
        # Read the CSV file data
        timestamps = []
        predictions = []
        
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                if len(row) >= 2:
                    try:
                        timestamps.append(float(row[0]))
                        predictions.append(row[1].strip())
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing row {row}: {e}")
        
        if timestamps and predictions:
            print(f"Loaded {len(timestamps)} data points from {csv_path}")
            # Create figures directory if it doesn't exist
            save_dir = os.path.dirname(csv_path)
            
            # Call function with loaded data
            question = "Is the bag getting pumped with air and squeezed by gloved fingers?"
            plot_binary_predictions_over_time(
                timestamps, 
                predictions, 
                "VLM and ROI", 
                question,
                class_name="Bag",
                save_dir=save_dir
            )
        else:
            print("No data was loaded from the CSV file or data is invalid.")
