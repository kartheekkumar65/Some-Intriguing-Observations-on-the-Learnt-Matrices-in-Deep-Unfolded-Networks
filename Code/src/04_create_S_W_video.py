import cv2
import glob
import os
from tqdm import tqdm

output_dir = './videos/histogram_videos_fps3'
video_filename = os.path.join(output_dir, 'test.mp4')  # Ensure this is the correct path and filename

# Fetch all the files matching the pattern and sort them to maintain the order
file_pattern = os.path.join(output_dir, 'S_W_constant_ep*.png')
files = sorted(glob.glob(file_pattern))

# Read the first image to determine the frame size
frame = cv2.imread(files[0])
height, width, layers = frame.shape
frame_size = (width, height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'x264' if you have it
fps = 3  # Frames per second
out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

# Loop through all images and write them to the video
for filename in tqdm(files):
    frame = cv2.imread(filename)
    out.write(frame)

# Release everything when job is finished
out.release()

print("Video creation completed.")