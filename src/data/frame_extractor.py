import os
from pathlib import Path

import cv2
import logging

logger = logging.getLogger(__name__)

def convert_videos_to_frames(videos_path: str, frames_path: str) -> None:

    video_paths = extract_file_names(videos_path)
    for video_path in video_paths:
        extract_frames(video_path, frames_path)

def extract_file_names(videos_path: str) -> list:
    """ Extract filenames from .lst file.

    Args:
        videos_path (str): Path to .lst file.

    Returns:
        list: List of filenames.
    """ 
    with open(videos_path, "r") as f:
        file_names = [line.strip() for line in f.readlines()]
    return file_names

def extract_frames(video_path: str, frame_path: str) -> None:
    
    count = 0
    vidcap = cv2.VideoCapture(video_path)
    
    success,image = vidcap.read()
    success = True
    logging.info(f"Extracting frames from {video_path}")

    file_name = Path(video_path).stem

    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = vidcap.read()
        if success:
            output_file = f"{frame_path}/{file_name}_{count}.jpg"
            logging.info(f"Writing to {output_file}")
            cv2.imwrite(output_file, image)     # save frame as JPEG file
            count = count + 1