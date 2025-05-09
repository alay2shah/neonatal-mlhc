"""
This script is used to load video data from a public GCS bucket and extract random frames from them.

brew install ffmpeg 

Usage: python dataloader/load_video_data.py --num-frames 5
"""

import os
import cv2
import numpy as np
import tempfile
from google.cloud import storage
import random
import argparse
from pathlib import Path
import subprocess
import shutil
import time
import io

class VideoDataLoader:
    def __init__(self, bucket_name="neonatal_video_data", timeout=30):
        """Initialize the VideoDataLoader with bucket info."""
        self.bucket_name = bucket_name
        self.timeout = timeout
        
        # Check if ffmpeg is installed
        if not shutil.which('ffmpeg'):
            raise RuntimeError("ffmpeg not found. Please install ffmpeg first.")
            
        # Initialize storage client for anonymous access
        self.storage_client = storage.Client.create_anonymous_client()
        self.bucket = self.storage_client.bucket(bucket_name)
        print("Successfully connected to public GCS bucket")
 
    def list_videos(self, prefix="videos/"):
        """List all videos in the specified bucket folder."""
        try:
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            videos = [blob for blob in blobs if blob.name.lower().endswith(('.mp4', '.avi', '.mov'))]
            if not videos:
                print(f"No videos found in prefix: {prefix}")
            else:
                print(f"Found {len(videos)} videos in bucket")
            return videos
        except Exception as e:
            print(f"Error listing videos: {str(e)}")
            return []

    def extract_random_frame_from_blob(self, blob):
        """Extract a random frame from a video blob using ffmpeg streaming."""
        print(f"\nProcessing video: {blob.name}")
        start_time = time.time()

        try:
            # Get video duration using ffprobe
            probe_command = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                f'https://storage.googleapis.com/{self.bucket_name}/{blob.name}'
            ]
            
            duration = float(subprocess.check_output(probe_command).decode().strip())
            random_time = random.uniform(0, min(duration, 30))  # First 30 seconds or full duration
            
            # Extract a single frame at the random timestamp
            frame_command = [
                'ffmpeg',
                '-ss', str(random_time),  # Seek to random position
                '-i', f'https://storage.googleapis.com/{self.bucket_name}/{blob.name}',
                '-frames:v', '1',  # Extract only one frame
                '-f', 'image2pipe',  # Output to pipe
                '-pix_fmt', 'bgr24',  # OpenCV format
                '-vcodec', 'rawvideo',  # Raw video format
                '-'  # Output to pipe
            ]
            
            # Run ffmpeg and capture output
            frame_data = subprocess.check_output(frame_command, timeout=self.timeout)
            
            # Convert to numpy array
            frame = np.frombuffer(frame_data, np.uint8)
            
            # Get frame dimensions from ffprobe
            size_command = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0',
                f'https://storage.googleapis.com/{self.bucket_name}/{blob.name}'
            ]
            
            dimensions = subprocess.check_output(size_command).decode().strip()
            width, height = map(int, dimensions.split('x'))
            
            # Reshape the frame data
            frame = frame.reshape((height, width, 3))
            
            print(f"Frame extracted in {time.time() - start_time:.1f}s")
            return frame

        except subprocess.TimeoutExpired:
            print(f"Timeout processing video {blob.name}")
            raise
        except Exception as e:
            print(f"Error processing video {blob.name}: {str(e)}")
            raise

    def load_random_frames(self, num_frames=5):
        """Load random frames from videos in the bucket."""
        videos = self.list_videos()
        if not videos:
            raise ValueError("No videos found in the specified bucket folder")

        all_frames = []
        video_sources = []
        attempts = 0
        max_attempts = num_frames * 2

        while len(all_frames) < num_frames and attempts < max_attempts:
            attempts += 1
            video_blob = random.choice(videos)
            try:
                frame = self.extract_random_frame_from_blob(video_blob)
                all_frames.append(frame)
                video_sources.append(video_blob.name)
                print(f"Successfully extracted frame {len(all_frames)} of {num_frames}")
                
            except Exception as e:
                print(f"Error processing video {video_blob.name}: {str(e)}")
                continue

        if not all_frames:
            raise ValueError("Failed to extract any frames after maximum attempts")
            
        return all_frames, video_sources

def main():
    """Example usage of the VideoDataLoader."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract random frames from videos in GCS bucket')
    parser.add_argument('--num-frames', type=int, default=150,
                      help='Total number of random frames to extract')
    parser.add_argument('--output-dir', type=str, 
                    #   default=str(Path.home() / 'neonatal-mlhc/testing_data'),
                        default=str(Path.cwd() / 'testing_data'),
                      help='Directory to save extracted frames')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = VideoDataLoader()
    frames, sources = loader.load_random_frames(num_frames=args.num_frames)
    
    print(f"Successfully loaded {len(frames)} frames")
    
    # Save frames with video source in filename
    for i, (frame, source) in enumerate(zip(frames, sources)):
        video_name = Path(source).stem
        filename = f"{video_name}_frame_{i}.jpg"
        filepath = output_dir / filename
        
        print(f"Saving frame {i} from video {source} to {filepath}")
        cv2.imwrite(str(filepath), frame)
    
    print(f"\nAll frames saved to: {output_dir}")

if __name__ == "__main__":
    main()
