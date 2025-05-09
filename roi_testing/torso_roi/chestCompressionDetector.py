# MAKE SURE TO pip install ultralytics 

import os
import cv2
import numpy as np
import csv
import time
from scipy.signal import find_peaks
from base64 import b64encode
from ultralytics import YOLO

class ChestCompressionDetector:
    def __init__(self, model_weights_path, compression_rate=90, window_size=60, output_interval=1, print_only_class=True):
        """
        Initialize the chest compression detector

        Args:
            model_weights_path: Path to YOLOv8 weights file
            compression_rate: Expected compressions per minute (default 90)
            window_size: Number of frames to analyze for rhythm detection
            output_interval: Output classification every N frames (default 1)
        """
        # Load the YOLO model with the provided weights
        self.model = YOLO(model_weights_path, verbose=False)

        # Parameters for chest compression detection
        self.expected_rate = compression_rate  # 90 compressions per minute
        self.window_size = window_size  # Frames to analyze for rhythm
        self.compression_history = []  # Store motion values for analysis
        self.output_interval = output_interval  # Print classification every N frames

        # Parameters for optical flow
        self.prev_gray = None
        self.stabilized_roi = None  # A fixed-size ROI for optical flow
        self.roi_padding = 10  # Padding around detection to maintain stable size

        # Initialize detection state
        self.compression_active = False
        self.roi = None  # Region of interest (where chest/thumbs are detected)

        # Initialize counters for state machine
        self.active_frame_count = 0
        self.inactive_frame_count = 0

        # For tracking compression duration
        self.continuous_compression_frames = 0
        self.min_compression_duration_frames = 0  # Will be set based on FPS
        self.fps = 30  # Default, will be updated with actual FPS
        self.valid_compression = False  # True only after minimum duration
        
        # Frame counter for regular output
        self.frame_counter = 0

        # Toggle if want classification print or other detailed motion info
        self.print_only_class = print_only_class


    def set_fps(self, fps):
        """Set the frames per second for timing calculations"""
        self.fps = fps
        # For 10 seconds at given FPS
        self.min_compression_duration_frames = int(10 * fps)
        if (not self.print_only_class):
            print(f"Set FPS to {fps}, minimum compression duration: {self.min_compression_duration_frames} frames")

    def set_output_interval(self, interval):
        """Set the interval (in frames) at which to output the binary classification"""
        self.output_interval = max(1, interval)  # Ensure at least 1

        if (not self.print_only_class):
            print(f"Set output interval to every {self.output_interval} frames")
            
    def output_binary_classification(self, force=False, print_only_classification=False):
        """
        Output the binary classification (0 or 1) based on current compression state
        
        Args:
            force: If True, output regardless of interval; if False, respect output_interval
            print_only_classification: If True, only print frame, timestamp and binary classification
        """
        if force or (self.frame_counter % self.output_interval == 0):
            binary_value = 1 if self.valid_compression else 0
            compression_seconds = self.continuous_compression_frames / self.fps if self.fps > 0 else 0
            
            if print_only_classification:
                # Calculate minutes and seconds for formatted timestamp
                total_seconds = self.frame_counter / self.fps if self.fps > 0 else 0
                minutes = int(total_seconds // 60)
                seconds = int(total_seconds % 60)
                # Print only frame number, timestamp and binary classification
                print(f"{self.frame_counter} {minutes:02d}:{seconds:02d} {binary_value}")
            else:
                print(f"FRAME {self.frame_counter}: Classification={binary_value}, Duration={compression_seconds:.1f}s")
                
            return binary_value
        return None

    def detect_roi(self, frame):
        """Detect the region of interest using YOLOv8"""
        try:
            results = self.model(frame, verbose = False)

            # Extract bounding boxes for the detected objects
            boxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    boxes.append((x1, y1, x2, y2, conf, cls))

            # If relevant objects are detected, define the ROI
            if boxes:
                # Take the highest confidence detection
                boxes.sort(key=lambda x: x[4], reverse=True)
                x1, y1, x2, y2, conf, cls = boxes[0]

                # Ensure ROI is within frame boundaries
                h, w = frame.shape[:2]
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)

                # Create a more stable ROI with padding
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                width = x2 - x1 + 2 * self.roi_padding
                height = y2 - y1 + 2 * self.roi_padding

                # Initialize stabilized ROI if first detection
                if self.stabilized_roi is None:
                    self.stabilized_roi = (width, height)
                else:
                    # Smooth transition to new size (avoid sudden changes)
                    stable_w, stable_h = self.stabilized_roi
                    width = int(0.9 * stable_w + 0.1 * width)
                    height = int(0.9 * stable_h + 0.1 * height)
                    self.stabilized_roi = (width, height)

                # Calculate fixed-size ROI around center
                half_width = width // 2
                half_height = height // 2
                x1 = max(0, center_x - half_width)
                y1 = max(0, center_y - half_height)
                x2 = min(w, center_x + half_width)
                y2 = min(h, center_y + half_height)

                # Ensure minimum ROI size
                if x2 - x1 < 10 or y2 - y1 < 10:
                    if self.roi is None:  # Only print once
                        print("ROI too small, using default region")
                    return False

                self.roi = (x1, y1, x2, y2)
                return True

            return False

        except Exception as e:
            print(f"Error in detect_roi: {e}")
            return False

    def compute_optical_flow(self, frame):
        """Compute optical flow in the ROI and return flow with visualization"""
        if self.roi is None:
            return False, None, None

        try:
            # Extract the ROI
            x1, y1, x2, y2 = self.roi
            roi_frame = frame[y1:y2, x1:x2]

            # Ensure ROI frame has valid dimensions
            if roi_frame.size == 0 or roi_frame.shape[0] < 2 or roi_frame.shape[1] < 2:
                print(f"Invalid ROI dimensions: {roi_frame.shape}")
                return False, None, None

            # Convert to grayscale
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

            # Apply fixed resize to ensure consistent dimensions for optical flow
            target_size = (150, 150)  # Fixed size for optical flow
            gray = cv2.resize(gray, target_size)

            # Initialize optical flow
            if self.prev_gray is None:
                self.prev_gray = gray
                return False, None, None

            # Calculate optical flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Create flow visualization
            flow_vis = self.draw_flow(gray, flow)

            # Update previous frame
            self.prev_gray = gray
            return True, flow, flow_vis

        except Exception as e:
            print(f"Error in compute_optical_flow: {e}")
            self.prev_gray = None  # Reset for next frame
            return False, None, None

    def draw_flow(self, img, flow, step=8):
        """Draw optical flow arrows on image"""
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y, x].T

        # Create visualization image
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Draw arrows
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)

        # Draw a background for better visibility of arrows
        cv2.rectangle(vis, (0, 0), (w, h), (64, 64, 64), -1)

        # Draw title
        cv2.putText(vis, "Optical Flow", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Color based on motion magnitude and direction
        for (x1, y1), (x2, y2) in lines:
            dx, dy = x2-x1, y2-y1
            magnitude = np.sqrt(dx*dx + dy*dy)

            # Skip very small motions
            if magnitude < 1:
                continue

            # Color based on direction (red for upward, blue for downward)
            if dy < 0:  # Upward motion (positive vertical in image coordinates is downward)
                color = (0, 0, 255)  # Red for upward (recoil)
            else:
                color = (255, 0, 0)  # Blue for downward (compression)

            cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 1, tipLength=0.3)

        # Draw legend
        cv2.putText(vis, "Compression", (5, h-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(vis, "Recoil", (5, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Draw a frame around the visualization
        cv2.rectangle(vis, (0, 0), (w-1, h-1), (255, 255, 255), 1)

        return vis

    def analyze_motion(self, flow):
        """Analyze motion to detect rhythmic compression patterns"""
        if flow is None:
            return 0, False

        try:
            # Calculate vertical motion (y-component more relevant for compressions)
            vertical_motion = np.mean(np.abs(flow[..., 1]))

            # Add to history
            self.compression_history.append(vertical_motion)

            # Keep history to a fixed window size
            if len(self.compression_history) > self.window_size:
                self.compression_history.pop(0)

            # If we have enough history, analyze for rhythmic patterns
            if len(self.compression_history) >= self.window_size // 2:  # Reduced requirement
                return self.detect_rhythm()

            return vertical_motion, False

        except Exception as e:
            print(f"Error in analyze_motion: {e}")
            return 0, False

    def detect_rhythm(self):
        """Detect if the motion follows a rhythmic pattern consistent with chest compressions"""
        try:
            # Get the motion history array
            motion_array = np.array(self.compression_history)

            # If not enough variation in motion, return false
            if np.max(motion_array) - np.min(motion_array) < 0.05:
                return motion_array[-1], False

            # Normalize the motion array for better peak detection
            norm_motion = (motion_array - np.min(motion_array)) / (np.max(motion_array) - np.min(motion_array) + 1e-6)

            # Parameters tuned for chest compression pattern
            # Find peaks in the motion (compressions)
            peaks, properties = find_peaks(
                norm_motion,
                height=0.3,       # Peak threshold
                distance=5,       # Minimum distance between peaks
                prominence=0.2    # Minimum prominence
            )

            # If we have enough peaks, calculate the rhythm
            if len(peaks) >= 3:  # Need at least 3 peaks to calculate rhythm
                # Calculate time between peaks
                peak_intervals = np.diff(peaks)
                avg_interval = np.mean(peak_intervals)

                # Assume 30 fps for rhythm calculation
                fps = self.fps
                compressions_per_minute = (fps * 60) / avg_interval

                # Check rhythm against expected rate (90 per min)
                is_rhythm = (
                    abs(compressions_per_minute - self.expected_rate) < 15 and  # Tolerance of 15 bpm
                    np.std(peak_intervals) / avg_interval < 0.3                # Consistency check
                )

                # Print debugging info occasionally
                if np.random.randint(0, 30) == 0:
                    if (not self.print_only_class):
                        print(f"Peaks: {len(peaks)}, Rate: {compressions_per_minute:.1f} bpm, " +
                            f"Consistency: {np.std(peak_intervals) / avg_interval:.2f}, Rhythm: {is_rhythm}")

                return motion_array[-1], is_rhythm

            return motion_array[-1], False

        except Exception as e:
            print(f"Error in detect_rhythm: {e}")
            return 0, False

    def process_frame(self, frame, print_only_classification=False):
        """Process a single frame to detect chest compressions"""
        try:
            # Increment frame counter
            self.frame_counter += 1
            
            # Make a copy of the frame to avoid modifying the original
            display_frame = frame.copy()

            # Check if ROI already exists, if not try to detect it
            if self.roi is None:
                roi_detected = self.detect_roi(frame)
                if not roi_detected:
                    # Reset compression state if ROI lost
                    prev_state = self.valid_compression
                    self.compression_active = False
                    self.continuous_compression_frames = 0
                    self.valid_compression = False
                    
                    # Output classification at regular intervals
                    self.output_binary_classification(print_only_classification=print_only_classification)
                    
                    return display_frame, False, 0

            # Periodically update ROI
            if np.random.randint(0, 30) == 0:
                self.detect_roi(frame)

            # Compute optical flow
            flow_success, flow, flow_vis = self.compute_optical_flow(frame)

            if not flow_success:
                # Reset compression state if flow calculation fails
                prev_state = self.valid_compression
                self.compression_active = False
                self.continuous_compression_frames = 0
                self.valid_compression = False
                
                # Output classification at regular intervals
                self.output_binary_classification(print_only_classification=print_only_classification)
                
                return display_frame, False, 0

            # Analyze motion to detect compressions
            motion_value, is_rhythm = self.analyze_motion(flow)

            # Fine-tuned thresholds based on testing
            high_threshold = 0.15   # Motion activation threshold
            low_threshold = 0.05    # Motion deactivation threshold

            # Update state counters with balanced logic
            if motion_value > high_threshold:
                # Increase active counter, faster if rhythm detected
                if is_rhythm:
                    self.active_frame_count += 2
                else:
                    self.active_frame_count += 1
                self.inactive_frame_count = max(0, self.inactive_frame_count - 1)
            elif motion_value < low_threshold:
                self.inactive_frame_count += 1
                self.active_frame_count = max(0, self.active_frame_count - 1)

            # Limit counters
            self.active_frame_count = min(10, self.active_frame_count)
            self.inactive_frame_count = min(10, self.inactive_frame_count)

            # Apply state change with appropriate thresholds
            required_active_frames = 4     # Frames required to activate
            required_inactive_frames = 6   # Frames required to deactivate

            # Store previous state for change detection
            prev_state = self.valid_compression
            prev_compression_active = self.compression_active

            # Update compression_active state
            if self.active_frame_count >= required_active_frames:
                self.compression_active = True
            elif self.inactive_frame_count >= required_inactive_frames:
                self.compression_active = False

            # Track continuous compression duration
            if self.compression_active:
                self.continuous_compression_frames += 1
            else:
                # Reset the counter when compression stops
                self.continuous_compression_frames = 0
                self.valid_compression = False

            # Check if we've reached the minimum duration (10 seconds)
            if self.continuous_compression_frames >= self.min_compression_duration_frames:
                self.valid_compression = True

            # Calculate seconds of continuous compression
            compression_seconds = self.continuous_compression_frames / self.fps
            
            # Output classification at regular intervals
            self.output_binary_classification(print_only_classification=print_only_classification)

            # Visualize on frame
            if self.roi:
                x1, y1, x2, y2 = self.roi

                # Different colors based on compression state and validity
                if self.valid_compression:
                    color = (0, 255, 0)  # Green for valid compression (>5 sec)
                elif self.compression_active:
                    color = (0, 255, 255)  # Yellow for active but not yet valid
                else:
                    color = (0, 0, 255)  # Red for inactive

                # Draw ROI rectangle
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                # Add optical flow visualization to top-right corner of frame
                if flow_vis is not None:
                    h, w = flow_vis.shape[:2]
                    # Position in top-right corner with 10px margin
                    display_frame[10:10+h, display_frame.shape[1]-w-10:display_frame.shape[1]-10] = flow_vis

                # Add text indicating compression status
                if self.valid_compression:
                    status = f"VALID COMPRESSION ({compression_seconds:.1f}s)"
                elif self.compression_active:
                    status = f"COMPRESSION ACTIVE ({compression_seconds:.1f}s / {10}.0s)"
                else:
                    status = "NO COMPRESSION"

                cv2.putText(display_frame, status, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Add motion value and counters
                cv2.putText(display_frame, f"Motion: {motion_value:.2f}", (x1, y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(display_frame, f"Active:{self.active_frame_count}/{required_active_frames} " +
                           f"Inactive:{self.inactive_frame_count}/{required_inactive_frames}",
                          (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Add binary classification value
                binary_value = 1 if self.valid_compression else 0
                cv2.putText(display_frame, f"Binary Classification: {binary_value}", (x1, y2+60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            return display_frame, self.valid_compression, motion_value

        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame, False, 0

    def process_video(self, video_path, output_path=None, csv_path=None, display_in_notebook=False, show_video=False, print_only_classification=False):
        """Process entire video and detect chest compressions

        Args:
            video_path: Path to input video file
            output_path: Path to save processed video (optional)
            csv_path: Path to save frame-by-frame classification CSV (optional)
            display_in_notebook: Whether to display processing in notebook (default: False)
            show_video: Whether to show video in a window during processing (default: False)
            print_only_classification: Whether to only print frame, timestamp, and binary classification (default: False)

        Returns:
            List of motion values and compression states
        """
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return [], []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return [], []

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if not print_only_classification:
            print(f"Video dimensions: {width}x{height}, FPS: {fps}")

        # Set FPS for duration calculations
        self.set_fps(fps)

        # Reset frame counter
        self.frame_counter = 0

        # Set up output video writer if needed
        writer = None
        if output_path:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # For plotting motion over time
        motion_values = []
        compression_states = []
        frame_numbers = []

        # For displaying in notebook
        display_id = None
        last_update = time.time()
        try:
            if display_in_notebook:
                # Try to import necessary libraries for notebook display
                try:
                    from IPython.display import display, HTML
                    # Create a placeholder for the output
                    display_id = display(HTML("Processing video..."), display_id=True)
                except ImportError:
                    if not print_only_classification:
                        print("Warning: IPython.display not available. Progress display disabled.")
                    display_in_notebook = False
        except Exception as e:
            if not print_only_classification:
                print(f"Warning: Could not set up notebook display: {e}")
            display_in_notebook = False

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not print_only_classification:
            print(f"Total frames: {total_frames}")

        # Prepare CSV file if path provided
        csv_writer = None
        csv_file = None
        if csv_path:
            try:
                # Ensure directory exists
                csv_dir = os.path.dirname(csv_path)
                if csv_dir and not os.path.exists(csv_dir):
                    os.makedirs(csv_dir, exist_ok=True)
                
                csv_file = open(csv_path, 'w', newline='')
                csv_writer = csv.writer(csv_file)
                # Write header
                csv_writer.writerow(['frame', 'timestamp', 'binary_classification', 'motion_value'])
            except Exception as e:
                print(f"Error creating CSV file: {e}")
                csv_writer = None

        # Process frames
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frame
                processed_frame, is_compression, motion_value = self.process_frame(frame, print_only_classification)

                # Store results for plotting and CSV output
                frame_numbers.append(frame_count)
                motion_values.append(motion_value)
                compression_states.append(1 if is_compression else 0)

                # Write to CSV if enabled
                if csv_writer:
                    timestamp = frame_count / fps
                    binary = 1 if is_compression else 0
                    csv_writer.writerow([frame_count, f"{timestamp:.3f}", binary, f"{motion_value:.4f}"])

                # Write to output video
                if writer:
                    writer.write(processed_frame)

                # Show video if requested
                if show_video:
                    cv2.imshow("Chest Compression Detection", processed_frame)
                    # Exit on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Display in notebook (update every 1 second to avoid slowdown)
                if display_in_notebook and display_id and time.time() - last_update > 1:
                    try:
                        from IPython.display import HTML
                        # Convert the frame to JPEG
                        _, buffer = cv2.imencode('.jpg', processed_frame)
                        # Convert to base64
                        b64_img = b64encode(buffer).decode('utf-8')
                        # Create an HTML img tag
                        img_html = f'<img src="data:image/jpeg;base64,{b64_img}" width="640" />'
                        # Update the display
                        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                        display_id.update(HTML(f'<p>Progress: {progress:.1f}%</p>{img_html}'))
                        last_update = time.time()
                    except Exception as e:
                        if not print_only_classification:
                            print(f"Warning: Could not update display: {e}")
                        display_in_notebook = False

                frame_count += 1

                # Print progress every 100 frames
                if frame_count % 100 == 0 and not print_only_classification:
                    print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        finally:
            # Release resources
            cap.release()
            if writer:
                writer.release()
            if csv_file:
                csv_file.close()
            if show_video:
                cv2.destroyAllWindows()

        # Plot the motion values and compression states
        if len(motion_values) > 0 and not print_only_classification:
            try:
                self.plot_results(motion_values, compression_states)
            except Exception as e:
                print(f"Warning: Could not plot results: {e}")

        if not print_only_classification:
            print(f"Processing complete!")
            if output_path:
                print(f"Processed video saved to {output_path}")
            if csv_path:
                print(f"Classification CSV saved to {csv_path}")

        return motion_values, compression_states

    def plot_results(self, motion_values, compression_states):
        """Plot the motion values and compression states"""
        try:
            # Import matplotlib only when needed
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))

            # Plot motion values
            plt.subplot(2, 1, 1)
            plt.plot(motion_values)
            plt.title('Vertical Motion')
            plt.ylabel('Motion Magnitude')
            plt.grid(True)

            # Highlight compression regions
            for i, is_active in enumerate(compression_states):
                if is_active:
                    plt.axvspan(i-0.5, i+0.5, color='green', alpha=0.3)

            # Plot compression states
            plt.subplot(2, 1, 2)
            plt.plot(compression_states)
            plt.title('Valid Compression Detection (>10 seconds)')
            plt.xlabel('Frame')
            plt.ylabel('Compression Active')
            plt.ylim(-0.1, 1.1)
            plt.grid(True)

            plt.tight_layout()

            # Try to save to local path
            try:
                plt.savefig('compression_analysis.png')
                print("Plot saved to: compression_analysis.png")
            except Exception as e:
                print(f"Couldn't save plot: {e}")

            # Try to display the plot if requested
            try:
                plt.show()
            except Exception as e:
                print(f"Couldn't display plot: {e}")

        except Exception as e:
            print(f"Error in plot_results: {e}")


# Main execution function
def main():
    # MAKE SURE THESE PARAMETERS ARE CORRECT

    # Detector parameters
    # Get weights from: https://www.dropbox.com/scl/fi/k6xli3tq9l0b7oa0gbuwo/best_weights_across_skin_color.pt?rlkey=wv0aw4wi1i71ipgrcvq26q0bg&st=c9vqyx6c&dl=0
    YOLO_MODEL_WEIGHTS_TORSO = "./best_weights_across_skin_color.pt" 
    COMPRESSION_RATE = 90 # ideal compression rate (compressions per minute)
    RHYTHM_WINDOW_SIZE = 60 #  Window size for motion analysis: number of frames used for rhythm detection; 
                            #  larger values (90+) provide more stable detection but slower response, 
                            #  smaller values (30-60) are more responsive to changes but potentially less stable
    OUTPUT_INTERVAL_FRAMES = 30 # For every how many frames to print out binary classification (0/1) of if chest compression is happening
    PRINT_ONLY_CLASSIFICATION = True # if false, also print out detailed motion info

    # Initialize detector with parameters
    detector = ChestCompressionDetector(
        model_weights_path=YOLO_MODEL_WEIGHTS_TORSO,
        compression_rate=COMPRESSION_RATE,
        window_size=RHYTHM_WINDOW_SIZE,
        output_interval=OUTPUT_INTERVAL_FRAMES,
    )

    # Video parameters
    VIDEO_NAME = "2.25.25_sc3_cropped_clip2" 
    INPUT_VIDEO_PATH = f"/Users/ejian/67930/stimulations/{VIDEO_NAME}.mp4"
    OUTPUT_VIDEO_PATH = f"/Users/ejian/67930/stimulations/{VIDEO_NAME}_output.mp4" 
    OUTPUT_CSV_PATH = f"/Users/ejian/67930/stimulations/{VIDEO_NAME}_output.csv" # csv that contains frame #, timestamp (in mm:ss format), binary classification (0/1)
    SHOW_VIDEO = True
    
    
    # Process the video
    print(f"Processing video file: {VIDEO_NAME}.mp4")
    detector.process_video(
        video_path=INPUT_VIDEO_PATH,
        output_path=OUTPUT_VIDEO_PATH,
        csv_path=OUTPUT_CSV_PATH,
        show_video=SHOW_VIDEO, 
        print_only_classification=PRINT_ONLY_CLASSIFICATION
    )

    # Note: process_video calls process_frame which calls detector.output_binary_classification to get binary classification (0/1) per frame
    # The output of detector.output_binary_classification already takes into account all necessary thresholds and does *not* need further processing.
    # However, it is important to note that a continuous compression must be maintained for a minimum of 10 seconds (min_compression_duration_frames = 10 * fps) 
    # before the detector outputs a positive classification (1). This means during testing the threshold added to the ground truth start timestamp of a chest compression 
    # should be >= 10. (e.g., if ground truth says that a chest compression starts at 0:05, then an accurate detector would only output 1's at 0:15, hypothetically.)
    #
    #
    # ######################################################################################################################
    #
    # Below is the detailed processing pipeline that is already done to determine if a frame is positive detection (1):
    #
    # 1) Continuous Duration Requirement: A continuous compression must be maintained for a minimum of 10 seconds 
    #    (min_compression_duration_frames = 10 * fps) before the detector outputs a positive classification (1).
    #    This prevents brief or random movements from triggering false positives.
    #
    # 2) Correct Compressions Per Minute (CPM): The system analyzes the rhythm of the detected motion by:
    #    - Finding peaks in the vertical motion using the scipy.signal.find_peaks function
    #    - Calculating the average interval between peaks to determine compressions per minute
    #    - Checking if this rate is within ±15 of the target rate (default 90 CPM)
    #    - Ensuring consistent rhythm by checking that the standard deviation of intervals is less than 30% of the mean
    #
    # 3) Motion Detection Thresholds: Multiple thresholds ensure accurate detection:
    #    - High threshold (0.15): Vertical motion must exceed this value to begin counting active frames
    #    - Low threshold (0.05): If motion falls below this, inactive frames are counted
    #    - Required active frames (4): At least 4 consecutive frames of high motion needed to activate compression state
    #    - Required inactive frames (6): At least 6 consecutive frames of low motion needed to deactivate compression state
    #
    # 4) Region of Interest (ROI) Detection: 
    #    - The YOLOv8 object detection model identifies the torso/chest area 
    #    - Optical flow analysis is applied only within this ROI to focus on relevant motion
    #    - ROI is automatically updated approximately every 30 frames to adapt to subject movement

if __name__ == "__main__":
    main()