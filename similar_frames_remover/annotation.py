from pydantic import BaseModel
from collections import defaultdict
from typing import List, Tuple, Callable, Dict
import datetime
import re
import os
import logging
import cv2
import numpy as np

class FrameAnnotation(BaseModel):
    image_name: str
    epoch_timestamp_ms: int
    hr_timestamp: str
    orig_img_size_wh: tuple[int, int]


def create_filename_parsers() -> List[Tuple[re.Pattern, Callable[[str], Tuple[str, int]]]]:
    def parse_epoch_format(filename: str):
        cam_id, ts = filename.split('-')
        return cam_id, int(ts)
    
    def parse_hr_format(filename: str):
        cam_id, date_time = filename.split('_', 1)
        timestamp_s = datetime.datetime.strptime(date_time, '%Y_%m_%d__%H_%M_%S').timestamp()
        return cam_id, int(timestamp_s*1000)

    parsers = [
        (re.compile('^c[0-9]*-[0-9]*$'), parse_epoch_format),
        (re.compile('^c[0-9]*_[0-9]{4}(_[0-9]{2}){2}_(_[0-9]{2}){3}$'), parse_hr_format)
    ]

    return parsers


def parse_filename(filename: str, parsers: List[Tuple[re.Pattern, Callable[[str], Tuple[str, int]]]]) -> Tuple[str | None, int | None]:
    basename = os.path.basename(filename)
    root, _ = os.path.splitext(basename)

    camera_id = None
    timestamp_ms = None

    for p in parsers:
        if p[0].fullmatch(root) is not None:
            camera_id, timestamp_ms = p[1](root)
            break

    return camera_id, timestamp_ms


def create_annotation(img_directory: str, min_img_side_size=300) -> Dict[str, List[FrameAnnotation]]:
    logging.info(f'Creating annotation for directory {img_directory}')
    sequences = defaultdict(list)

    filename_parsers = create_filename_parsers()

    for fname in os.listdir(img_directory):
        img = cv2.imread(os.path.join(img_directory, fname))

        if img is None:
            # issue with image reading
            logging.info(f'File {fname} is skipped due to issues with image parsing')
            continue

        img_size_wh = img.shape[1::-1]
        if np.any(np.asarray(img_size_wh) < min_img_side_size):
            # the image is too small
            logging.info(f'Image {fname} is skipped due to small size min({img_size_wh}) < {min_img_side_size}')
            continue

        cam_id, timestamp_ms = parse_filename(fname, filename_parsers)

        if cam_id is None or timestamp_ms is None:
            # issue with parsing the filename
            logging.info(f'Image {fname} is skipped due to incompatible file name format')
            continue
        
        hr_timestamp = datetime.datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y_%m_%d__%H_%M_%S')
        sequences[cam_id].append(FrameAnnotation(image_name=fname, epoch_timestamp_ms=timestamp_ms, 
                                                 hr_timestamp=hr_timestamp, orig_img_size_wh=img_size_wh))

    for cam_id in sequences.keys():
        sequences[cam_id] = list(sorted(sequences[cam_id], key=lambda s: s.epoch_timestamp_ms))

    logging.info(f'Annotation is created, num samples per sequences: {[f"{cam_id}: {len(sequences[cam_id])}" for cam_id in sorted(sequences.keys())]}')
    return sequences