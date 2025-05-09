import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import torch
from models import YOLOv8Detector, track_with_lk_optical_flow

class BagSqueezeDetector:
    def __init__(self, model_path, squeeze_rate=40, window_size=15):
        """
        Initialize the bag squeeze detector with parameters optimized for pumping/contortion detection
        
        Args:
            model_path: Path to YOLOv8 weights file for bag detection
            squeeze_rate: Expected squeezes per minute (default now 60, approximately one per second)
            window_size: Number of frames to analyze for rhythm detection
        """
        # Load the YOLO model
        self.detector = YOLOv8Detector(model_path)
        
        # Parameters for squeeze detection
        self.expected_rate = squeeze_rate  # 60 squeezes per minute (one per second)
        self.window_size = window_size  # Frames to analyze for rhythm
        self.motion_history = []  # Store motion values for analysis
        
        # Parameters for optical flow
        self.prev_gray = None
        self.prev_roi_gray = None  # Add specific storage for ROI gray frame
        self.roi = None  # Region of interest (where bag is detected)
        
        # Initialize detection state
        self.squeeze_active = False
        self.last_squeeze_frame = 0
        self.min_frames_between_squeezes = 5  # Will be updated based on FPS
        
        # For tracking compressions
        self.squeeze_count = 0
        self.squeeze_timestamps = []
        self.fps = 30  # Default, will be updated
        
        # Add tracking parameters
        self.detection_interval = 15  # Run detection every 15 frames
        self.tracked_boxes = []  # Track boxes between detections
        
        # Add parameters for continuous squeeze detection (optimized for pumping motions)
        self.potential_squeeze_start = 0  # Frame when a potential squeeze began
        self.continuous_squeeze_frames = 0  # Counter for continuous squeeze frames
        self.squeeze_duration_threshold = 0.2  # Shorter duration threshold for quick pumps
        self.squeeze_in_progress = True  # Tracking if we're in the middle of detecting a squeeze
        self.motion_threshold = 15.0  # Threshold for complex motions
        self.inactivity_counter = 0  # Counter for tracking inactivity after a squeeze
        self.inactivity_threshold = 60  # How many frames of inactivity to end sustained detection
        self.sustained_detection = True  # Flag for sustained detection state
        self.max_sustained_duration = 1  # Maximum duration (seconds) without new motion
        
    def set_fps(self, fps):
        """Set the frames per second for timing calculations and adjust thresholds accordingly"""
        self.fps = fps
        
        # Adjust timing parameters based on framerate
        # For 60 squeezes per minute (1 per second)
        self.min_frames_between_squeezes = max(3, int(fps * 0.05))  # Allow new squeeze after 0.05 seconds
        self.inactivity_threshold = int(fps * 0.2)  # 0.2 seconds of inactivity to end sustained detection
        
        print(f"Set FPS to {fps}, adjusted parameters:")
        print(f"  - Min frames between squeezes: {self.min_frames_between_squeezes}")
        print(f"  - Inactivity threshold: {self.inactivity_threshold} frames")
    
    def detect_roi(self, frame):
        """Detect the bag using YOLOv8 with increased padding"""
        detections = self.detector.detect(frame, conf_threshold=0.25)
        padding = 0.1
        # Look for bag (class_id = 0)
        bags = [box for box in detections if int(box[5]) == 0]
        
        if bags:
            # Sort by confidence
            bags.sort(key=lambda x: x[4], reverse=True)
            x1, y1, x2, y2, conf, _ = bags[0]
            
            # Apply more generous padding to ROI (increased from 0.15 to 0.25)
            h, w = frame.shape[:2]
            roi_width = x2 - x1
            roi_height = y2 - y1
            padding_x = int(roi_width * padding)  # Increased padding
            padding_y = int(roi_height * padding)  # Increased padding
            
            # Apply padding with boundary checks
            padded_x1 = max(0, int(x1) - padding_x)
            padded_y1 = max(0, int(y1) - padding_y)
            padded_x2 = min(w, int(x2) + padding_x)
            padded_y2 = min(h, int(y2) + padding_y)
            
            self.roi = (padded_x1, padded_y1, padded_x2, padded_y2)
            return True, self.roi, conf
            
        return False, None, 0
    
    def compute_optical_flow(self, frame):
        """Compute optical flow in the ROI to detect squeezing motion"""
        if self.roi is None:
            return False, None, None
            
        # Extract ROI
        x1, y1, x2, y2 = self.roi
        roi_frame = frame[y1:y2, x1:x2]
        
        # Ensure ROI is valid
        if roi_frame.size == 0 or roi_frame.shape[0] < 5 or roi_frame.shape[1] < 5:
            return False, None, None
            
        # Convert to grayscale
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Normalize size for consistent processing
        target_size = (150, 150)
        gray = cv2.resize(gray, target_size)
        
        # Initialize or update optical flow
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
            
    def draw_flow(self, img, flow, step=8):
        """Draw optical flow arrows with better visualization for contorted/pumping motions"""
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y, x].T
        
        # Create visualization image
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw arrows
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        
        # Draw a background
        cv2.rectangle(vis, (0, 0), (w, h), (64, 64, 64), -1)
        
        # Draw title
        cv2.putText(vis, "Bag Motion", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Count inward and outward motion vectors
        inward_count = 0
        outward_count = 0
        total_count = 0
        
        # Color based on motion direction and calculate inward/outward ratio
        for (x1, y1), (x2, y2) in lines:
            dx, dy = x2-x1, y2-y1
            magnitude = np.sqrt(dx*dx + dy*dy)
            
            # Skip very small motions
            if magnitude < 1:
                continue
            
            total_count += 1
                
            # Check if motion is inward from sides (x direction)
            # Left side moving right OR right side moving left
            is_inward_x = (x1 < w/2 and dx > 0) or (x1 > w/2 and dx < 0)
            
            if is_inward_x:
                color = (0, 0, 255)  # Red for compression/squeeze
                inward_count += 1
            else:
                color = (0, 255, 0)  # Green for expansion
                outward_count += 1
                
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 1, tipLength=0.3)
        
        # Draw inward/outward ratio
        if total_count > 0:
            inward_ratio = inward_count / total_count
            cv2.putText(vis, f"Inward: {inward_ratio:.2f}", (5, h-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw legend
        cv2.putText(vis, "Squeeze", (5, h-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(vis, "Expand", (5, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                   
        # Frame the visualization
        cv2.rectangle(vis, (0, 0), (w-1, h-1), (255, 255, 255), 1)
        
        return vis
        
    def analyze_motion(self, flow, frame_idx):
        """Analyze flow to detect pumping/squeezing motions with better contortion detection"""
        if flow is None:
            # Reset continuous squeeze counter when no flow is available
            self.continuous_squeeze_frames = 0
            self.squeeze_in_progress = False
            self.sustained_detection = False
            self.inactivity_counter = 0
            return 0, False
            
        # Calculate overall motion in the ROI (not just directional)
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_motion = np.mean(flow_magnitude)
        
        # Calculate motion complexity - variance in different directions
        # Higher complexity indicates contortion/random movements rather than uniform motion
        flow_complexity = np.std(flow_magnitude) 
        
        # Calculate directional components
        horizontal_motion = np.mean(np.abs(flow[..., 0]))
        vertical_motion = np.mean(np.abs(flow[..., 1]))
        
        # Calculate directional changes - higher values mean more changing directions
        h, w = flow.shape[:2]
        center_region = flow[h//4:3*h//4, w//4:3*w//4]  # Focus on center
        
        # Check for opposing directions in the flow (indicative of pumping)
        left_region = flow[:, :w//2]
        right_region = flow[:, w//2:]
        left_x_motion = np.mean(left_region[..., 0])
        right_x_motion = np.mean(right_region[..., 0])
        
        # Detect opposing horizontal movements (classic pumping pattern)
        opposing_motion = (left_x_motion * right_x_motion < 0)
        
        # Calculate a pumping score that combines overall motion, complexity and opposing directions
        pumping_score = (mean_motion * 3.0 + 
                        flow_complexity * 2.0 + 
                        (horizontal_motion + vertical_motion) * 1.0)
        
        if opposing_motion:
            pumping_score *= 1.5  # Boost score if we detect opposing motions
            
        # Store this comprehensive motion value
        self.motion_history.append(pumping_score)
        
        # Keep history to fixed window size
        if len(self.motion_history) > self.window_size:
            self.motion_history.pop(0)
            
        # Detect squeezing using adjusted thresholds for pumping motions
        squeeze_detected = False
        
        # Only analyze if we have enough history
        if len(self.motion_history) >= 5:
            # Adjust threshold for pumping detection
            motion_threshold = self.motion_threshold 
            
            # Look for any significant motion with complexity
            is_potential_squeeze = pumping_score > motion_threshold
            
            # Handle sustained detection
            if self.sustained_detection:
                # End sustained detection when motion drops significantly
                if pumping_score < motion_threshold * 0.35:  # Lower threshold for ending sustained detection
                    self.inactivity_counter += 2
                    
                    # Very aggressive inactivity detection
                    inactivity_threshold = min(self.inactivity_threshold, 2)
                    
                    if self.inactivity_counter >= inactivity_threshold:
                        print(f"ENDING sustained detection - motion too low: {pumping_score:.4f}")
                        self.sustained_detection = False
                        self.inactivity_counter = 0
                        return pumping_score, False
                else:
                    self.inactivity_counter = 0
                    
                # While in sustained detection, continue reporting squeeze as active
                return pumping_score, True
            
            # Handle potential squeeze detection
            if is_potential_squeeze:
                if not self.squeeze_in_progress:
                    self.potential_squeeze_start = frame_idx
                    self.squeeze_in_progress = True
                    self.continuous_squeeze_frames = 1
                else:
                    self.continuous_squeeze_frames += 1
                
                # Check if we've reached the duration threshold
                squeeze_duration = self.continuous_squeeze_frames / self.fps
                if squeeze_duration >= self.squeeze_duration_threshold:
                    if frame_idx - self.last_squeeze_frame > self.min_frames_between_squeezes:
                        squeeze_detected = True
                        self.last_squeeze_frame = frame_idx
                        self.squeeze_count += 1
                        self.squeeze_timestamps.append(frame_idx / self.fps)
                        
                        # Enter sustained detection mode
                        self.sustained_detection = True
                        self.inactivity_counter = 0
            else:
                # Reset if motion stops
                self.squeeze_in_progress = False
                self.continuous_squeeze_frames = 0
        
        return pumping_score, squeeze_detected or self.sustained_detection
    
    def process_frame(self, frame, frame_idx):
        """Process a single frame to detect bag and squeezing motion"""
        # Make a copy for display
        display_frame = frame.copy()
        
        # Convert frame to grayscale for optical flow
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Run YOLO detection at specified intervals or when we have no boxes to track
        if frame_idx % self.detection_interval == 0 or not self.tracked_boxes:
            # Detect objects in the frame
            detections = self.detector.detect(frame, conf_threshold=0.25)
            
            # Filter for bag detections (class_id = 0)
            self.tracked_boxes = [box for box in detections if int(box[5]) == 0]
            
            # Reset optical flow if we have a new detection
            self.prev_gray = curr_gray
        else:
            # Use optical flow tracking between detections
            if self.prev_gray is not None and self.tracked_boxes:
                self.tracked_boxes = track_with_lk_optical_flow(self.prev_gray, curr_gray, self.tracked_boxes)
        
        # Update previous frame for next optical flow calculation
        self.prev_gray = curr_gray
        
        # Process the best bag detection if available
        if self.tracked_boxes:
            # Sort by confidence if available (for newly detected boxes)
            if len(self.tracked_boxes[0]) > 5:  # Check if confidence is present
                self.tracked_boxes.sort(key=lambda x: x[4] if len(x) > 4 else 0, reverse=True)
            
            # Get the highest confidence bag
            best_box = self.tracked_boxes[0]
            x1, y1, x2, y2 = best_box[:4]
            
            # Apply padding to ROI
            h, w = frame.shape[:2]
            roi_width = x2 - x1
            roi_height = y2 - y1
            padding_x = int(roi_width * 0.15)
            padding_y = int(roi_height * 0.15)
            
            # Apply padding with boundary checks
            padded_x1 = max(0, int(x1) - padding_x)
            padded_y1 = max(0, int(y1) - padding_y)
            padded_x2 = min(w, int(x2) + padding_x)
            padded_y2 = min(h, int(y2) + padding_y)
            
            # Update the ROI
            self.roi = (padded_x1, padded_y1, padded_x2, padded_y2)
            
            # Extract ROI for optical flow analysis
            roi_frame = frame[padded_y1:padded_y2, padded_x1:padded_x2]
            
            # Ensure ROI is valid
            if roi_frame.size > 0 and roi_frame.shape[0] >= 5 and roi_frame.shape[1] >= 5:
                # Convert to grayscale
                roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                
                # Normalize size for consistent processing
                target_size = (150, 150)
                roi_gray = cv2.resize(roi_gray, target_size)
                
                # Calculate optical flow if we have a previous ROI frame
                flow = None
                flow_vis = None
                motion_value = 0
                
                # Store current ROI for next iteration if we don't have one yet
                if not hasattr(self, 'prev_roi_gray') or self.prev_roi_gray is None:
                    self.prev_roi_gray = roi_gray
                else:
                    # Calculate optical flow (Farneback)
                    flow = cv2.calcOpticalFlowFarneback(
                        self.prev_roi_gray, roi_gray,
                        None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    
                    # Create flow visualization
                    flow_vis = self.draw_flow(roi_gray, flow)
                    
                    # Update previous ROI frame
                    self.prev_roi_gray = roi_gray
                    
                    # Analyze motion for squeeze detection
                    motion_value, squeeze_detected = self.analyze_motion(flow, frame_idx)
                    
                    # Update the squeeze state
                    self.squeeze_active = squeeze_detected
                    
                    # Calculate squeeze rate (squeezes per minute)
                    squeeze_rate = 0
                    if len(self.squeeze_timestamps) >= 2:
                        # Calculate from recent squeezes (last 5 or all if less than 5)
                        recent_timestamps = self.squeeze_timestamps[-5:] if len(self.squeeze_timestamps) > 5 else self.squeeze_timestamps
                        intervals = np.diff(recent_timestamps)
                        if len(intervals) > 0:
                            avg_interval = np.mean(intervals)
                            if avg_interval > 0:
                                squeeze_rate = 60.0 / avg_interval  # Convert to per minute
                    
                    # Visualize on frame
                    for box in self.tracked_boxes:
                        x1, y1, x2, y2 = box[:4]
                        
                        # Color based on squeeze state
                        color = (0, 255, 0) if self.squeeze_active else (0, 165, 255)  # Green if active, orange otherwise
                        
                        # Draw ROI rectangle
                        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add optical flow visualization to corner
                        if flow_vis is not None:
                            h, w = flow_vis.shape[:2]
                            # Position in top-right with margin
                            display_frame[10:10+h, display_frame.shape[1]-w-10:display_frame.shape[1]-10] = flow_vis
                            # Add squeeze duration information
                            if self.squeeze_in_progress:
                                current_duration = self.continuous_squeeze_frames / self.fps
                                duration_text = f"Squeeze duration: {current_duration:.1f}s / {self.squeeze_duration_threshold:.1f}s"
                                duration_color = (0, 165, 255)  # Orange until threshold met
                                
                                if current_duration >= self.squeeze_duration_threshold:
                                    duration_color = (0, 255, 0)  # Green when threshold met
                                
                                cv2.putText(display_frame, duration_text, (int(x1), int(y2)+100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, duration_color, 2)
                            
                            # Show squeeze progress bar if in progress
                            if self.squeeze_in_progress:
                                progress = min(1.0, self.continuous_squeeze_frames / (self.squeeze_duration_threshold * self.fps))
                                bar_width = 150
                                bar_height = 10
                                bar_x = int(x1)
                                bar_y = int(y2) + 115
                                
                                # Draw background bar
                                cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (64, 64, 64), -1)
                                
                                # Draw progress
                                progress_width = int(bar_width * progress)
                                cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                                            (0, 165, 255) if progress < 1.0 else (0, 255, 0), -1)
                                        
                                # Draw progress percentage
                                cv2.putText(display_frame, f"Progress: {int(progress * 100)}%", 
                                            (bar_x + 5, bar_y + bar_height + 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                        # Show squeeze status
                        status = "SQUEEZE DETECTED" if self.squeeze_active else "MONITORING"
                        cv2.putText(display_frame, status, (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                        # Show motion value and count
                        cv2.putText(display_frame, f"Motion: {motion_value:.2f}", (int(x1), int(y2)+25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.putText(display_frame, f"Squeeze count: {self.squeeze_count}", (int(x1), int(y2)+50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.putText(display_frame, f"Rate: {squeeze_rate:.1f}/min", (int(x1), int(y2)+75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        # Add a new status line for sustained detection to the display
                        if hasattr(self, 'sustained_detection') and self.sustained_detection:
                            cv2.putText(
                                display_frame, 
                                f"SUSTAINED DETECTION (motion: {motion_value:.2f}, inactive: {self.inactivity_counter}/{self.inactivity_threshold})",
                                (int(self.roi[0]), int(self.roi[1])-30),  # Position above the ROI
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                            )
                    
                    return display_frame, self.squeeze_active, motion_value, flow_vis
        
        # Default return if no ROI or optical flow
        return display_frame, False, 0, None
    
    def process_video(self, video_path, output_path=None, display=True, end_time=None):
        """Process a video to detect bag squeezing"""
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.set_fps(fps)
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        # For plotting
        motion_values = []
        squeeze_states = []
        frame_indices = []
        
        # Process each frame
        frame_idx = 0
        max_frames = total_frames
        if end_time is not None:
            max_frames = min(total_frames, int(end_time * fps))
            
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame, is_squeeze, motion_value, _ = self.process_frame(frame, frame_idx)
            
            # Store for plotting
            frame_indices.append(frame_idx)
            motion_values.append(motion_value)
            squeeze_states.append(1 if is_squeeze else 0)
            
            # Write to output
            if writer:
                writer.write(processed_frame)
                
            # Display if requested
            if display:
                cv2.imshow('Bag Squeeze Detection', processed_frame)
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):  # ESC or q
                    break
            
            # Update progress
            if frame_idx % 100 == 0:
                print(f"Progress: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
                
            frame_idx += 1
            
        # Clean up
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Plot results
        self.plot_results(frame_indices, motion_values, squeeze_states, fps)
        
        return motion_values, squeeze_states
        
    def plot_results(self, frame_indices, motion_values, squeeze_states, fps):
        """Plot the motion values and squeeze detections"""
        plt.figure(figsize=(12, 8))
        
        # Create time axis in seconds
        times = [f/fps for f in frame_indices]
        
        # Plot motion values
        plt.subplot(2, 1, 1)
        plt.plot(times, motion_values)
        plt.title('Bag Motion Magnitude')
        plt.ylabel('Motion Value')
        plt.grid(True)
        
        # Highlight squeeze regions
        for i, is_squeeze in enumerate(squeeze_states):
            if is_squeeze:
                plt.axvspan(times[i]-0.1, times[i]+0.1, color='red', alpha=0.3)
                
        # Plot squeeze detections
        plt.subplot(2, 1, 2)
        plt.plot(times, squeeze_states)
        plt.title('Squeeze Detection')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Squeeze Detected')
        plt.ylim(-0.1, 1.1)
        plt.grid(True)
        
        # Mark squeeze timestamps
        for ts in self.squeeze_timestamps:
            plt.axvline(x=ts, color='r', linestyle='--', alpha=0.7)
            
        # Calculate and display squeeze rate statistics
        if self.squeeze_timestamps:
            intervals = np.diff(self.squeeze_timestamps)
            if len(intervals) > 0:
                avg_interval = np.mean(intervals)
                rate_per_min = 60.0 / avg_interval if avg_interval > 0 else 0
                plt.figtext(0.5, 0.01, 
                          f"Total squeezes: {self.squeeze_count}, Avg interval: {avg_interval:.2f}s, Rate: {rate_per_min:.1f}/min",
                          ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = "squeeze_detection_output"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "bag_squeeze_analysis.png"))
        plt.show()

# Example usage
if __name__ == "__main__":
    # Paths should be updated to your environment
    model_path = "./runs/detect/train/weights/best.pt"  # YOLOv8 weights for bag detection
    video_path = "./dataset/2.6.25_sc8.mp4"  # Video to analyze
    output_path = "./squeeze_detection_output/output_video.mp4"  # Output path
    
    # Create detector
    detector = BagSqueezeDetector(model_path)
    
    # Process video
    detector.process_video(video_path, output_path, display=True)
