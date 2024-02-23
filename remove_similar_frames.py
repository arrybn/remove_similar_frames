from ast import Call, Dict, Tuple
import os
from posixpath import basename
from pydantic import BaseModel
from collections import defaultdict
import re
import datetime
from typing import Iterable, List, Tuple, Callable, Dict, Union
import cv2
import numpy as np
import copy
from thirdrdparty.imaging_interview import preprocess_image_change_detection, compare_frames_change_detection
from functools import lru_cache
from itertools import compress


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
    sequences = defaultdict(list)

    filename_parsers = create_filename_parsers()

    for fname in os.listdir(img_directory):
        img = cv2.imread(os.path.join(img_directory, fname))

        if img is None:
            # issue with image reading
            continue

        img_size_wh = img.shape[1::-1]
        if np.any(np.asarray(img_size_wh) < min_img_side_size):
            # the image is too small
            continue

        cam_id, timestamp_ms = parse_filename(fname, filename_parsers)

        if cam_id is None or timestamp_ms is None:
            # issue with parsing the filename
            continue
        
        hr_timestamp = datetime.datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y_%m_%d__%H_%M_%S')
        sequences[cam_id].append(FrameAnnotation(image_name=fname, epoch_timestamp_ms=timestamp_ms, 
                                                 hr_timestamp=hr_timestamp, orig_img_size_wh=img_size_wh))

    for cam_id in sequences.keys():
        sequences[cam_id] = list(sorted(sequences[cam_id], key=lambda s: s.epoch_timestamp_ms))

    return sequences


def get_gaussian_blur_ksize(img_size_wh: tuple[int, int], gaussian_blur_ksize_ratio: Union[float, None]):
    gaussian_blur_ksize = None
    if gaussian_blur_ksize_ratio is not None and gaussian_blur_ksize_ratio > 0.0:
        gaussian_blur_ksize = [max(2*int(min(img_size_wh) * gaussian_blur_ksize_ratio) + 1, 3)]

    return gaussian_blur_ksize


def preprocess_imgs(imgs: List[np.ndarray], gblur_half_size_ratio: Union[float, None]=0.025) -> List[np.ndarray]:
    imgs_shapes = np.asarray([i.shape[:-1] for i in imgs])
    
    preprocessed_imgs = []

    if not (imgs_shapes == imgs_shapes[0]).all():
        areas = imgs_shapes[:, 0] * imgs_shapes[:, 1]
        shape_to_resize = imgs_shapes[np.argmin(areas)]
        preprocessed_imgs = [cv2.resize(i, shape_to_resize[::-1]) if not (i.shape[:-1] == shape_to_resize).all() else i for i in imgs]
    else:
        preprocessed_imgs = copy.deepcopy(imgs)


    gaussian_blur_ksize = get_gaussian_blur_ksize(imgs_shapes[0], gblur_half_size_ratio)
    # if gblur_half_size_ratio is not None and gblur_half_size_ratio > 0.0:
        # gaussian_blur_ksize = [max(2*int(.min()*gblur_half_size_ratio) + 1, 3)]

    preprocessed_imgs = [preprocess_image_change_detection(i, gaussian_blur_ksize) for i in preprocessed_imgs]

    return preprocessed_imgs


def create_caching_load_preprocess(cache_size=None) -> Callable[[str, str, tuple[int, int], Union[float, None]], np.ndarray]:
    @lru_cache(maxsize=cache_size)
    def load_preprocess(img_name: str, dataset_path: str, out_img_size_wh: tuple[int, int], 
                        gaussian_blur_ksize_ratio: Union[float, None]) -> np.ndarray:
        img = cv2.imread(os.path.join(dataset_path, img_name), cv2.IMREAD_COLOR)

        img = cv2.resize(img, out_img_size_wh)
        gaussian_blur_ksize = get_gaussian_blur_ksize(out_img_size_wh, gaussian_blur_ksize_ratio)

        return preprocess_image_change_detection(img, gaussian_blur_ksize)
    
    return load_preprocess


def compare_imgs(img1: np.ndarray, img2: np.ndarray, min_cnt_area_ratio: float):
    imgs_area = np.prod(img1.shape[1::-1])
    score_nonnormed, contours, thresholded_diff = compare_frames_change_detection(img1, img2, 
                                                                                  imgs_area*min_cnt_area_ratio)
    score = score_nonnormed
    if len(contours) != 0:
        score = score_nonnormed / imgs_area

    return score, contours, thresholded_diff


def are_imgs_similar(img1: np.ndarray, img2: np.ndarray, min_cnt_area_ratio: float, score_threshold: float) -> bool:
    score, _, _ = compare_imgs(img1, img2, min_cnt_area_ratio)

    return score < score_threshold


def get_min_area_img_size(img_seq: List[FrameAnnotation]) -> tuple[int, int]:
    min_area_img_sample = min(img_seq, key=lambda sample: np.prod(sample.orig_img_size_wh))
    out_img_size_wh = min_area_img_sample.orig_img_size_wh

    return out_img_size_wh


def remove_subsequent_similar_frames(img_seq: List[FrameAnnotation], dataset_path: str, resize_img_size_wh: tuple[int, int],
                                     load_imgs_function: Callable[[str, str, tuple[int, int], Union[float, None]], np.ndarray], 
                                     gaussian_blur_ksize_ratio: Union[float, None], 
                                     min_cnt_area_ratio: float,
                                     score_threshold: float) -> List[FrameAnnotation]:
    similarity_mask = [True]

    for prev_sample, next_sample in zip(img_seq, img_seq[1:]):
        assert next_sample.epoch_timestamp_ms >= prev_sample.epoch_timestamp_ms

        similar = are_imgs_similar(load_imgs_function(prev_sample.image_name, 
                                                      dataset_path, resize_img_size_wh, 
                                                      gaussian_blur_ksize_ratio),
                                    load_imgs_function(next_sample.image_name, 
                                                       dataset_path, resize_img_size_wh, 
                                                       gaussian_blur_ksize_ratio),
                                    min_cnt_area_ratio, score_threshold)
        
        similarity_mask.append(not similar)

    unique_sebsequent_frames = list(compress(img_seq, similarity_mask))

    return unique_sebsequent_frames

    