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
from sklearn.cluster import DBSCAN
import argparse
import logging


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
    logging.info(f'Remove subsequent similar frames')
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

    logging.info(f'Removed {len(img_seq) - len(unique_sebsequent_frames)} frames from {len(img_seq)}')

    return unique_sebsequent_frames


def remove_similar_frames_by_clustering(img_seq: List[FrameAnnotation],
                                        dataset_path: str, resize_img_size_wh: tuple[int, int],
                                        load_imgs_function: Callable[[str, str, tuple[int, int], 
                                                                      Union[float, None]], np.ndarray],
                                        gaussian_blur_ksize_ratio: Union[float, None], 
                                        min_cnt_area_ratio: float,
                                        score_threshold: float) -> List[FrameAnnotation]:
    logging.info(f'Removing similar frames by clustering')

    indices = np.asarray(list(range(len(img_seq)))).reshape((-1, 1))

    def metric(index_1, index_2):
        sample_1 = img_seq[int(index_1.item())]
        sample_2 = img_seq[int(index_2.item())]
        
        img1 = load_imgs_function(sample_1.image_name, dataset_path, resize_img_size_wh, gaussian_blur_ksize_ratio)
        img2 = load_imgs_function(sample_2.image_name, dataset_path, resize_img_size_wh, gaussian_blur_ksize_ratio)

        similarity, _, _ = compare_imgs(img1, img2, min_cnt_area_ratio)
        
        return similarity
    
    clustering = DBSCAN(eps=score_threshold, min_samples=1, metric=metric).fit(indices)

    unique_images_annotation = list(dict(zip(clustering.labels_, img_seq)).values())

    logging.info(f'Removed {len(img_seq) - len(unique_images_annotation)} frames from {len(img_seq)}')

    return unique_images_annotation


class SimilarImagesRemover:
    def __init__(self, gaussian_blur_hksize_ratio: float, min_cnt_area_ratio: float,
                 similar_score_thr: float, cache_size: Union[int, None], min_img_size_to_process) -> None:
        self.gaussian_blur_hksize_ratio = gaussian_blur_hksize_ratio
        self.min_cnt_area_ratio = min_cnt_area_ratio
        self.similar_score_thr = similar_score_thr
        self.min_img_size_to_process = min_img_size_to_process

        self.load_prep_func = create_caching_load_preprocess(cache_size)

    def process_directory(self, imgs_dir_path):
        logging.info(f'Processing direcroty: {imgs_dir_path}')
        full_annotation = create_annotation(imgs_dir_path, self.min_img_size_to_process)

        unique_frames = []

        for cam_id in sorted(full_annotation.keys()):
            logging.info(f'Processing sequence for camera: {cam_id}')
            img_size = get_min_area_img_size(full_annotation[cam_id])

            subseq_unique_samples = remove_subsequent_similar_frames(full_annotation[cam_id], imgs_dir_path, 
                                                         img_size, self.load_prep_func, self.gaussian_blur_hksize_ratio, 
                                                         self.min_cnt_area_ratio, self.similar_score_thr)
            
            cam_unique_frames = remove_similar_frames_by_clustering(subseq_unique_samples, imgs_dir_path, img_size, 
                                                                self.load_prep_func, self.gaussian_blur_hksize_ratio, 
                                                                self.min_cnt_area_ratio, self.similar_score_thr)
            
            unique_frames.extend(cam_unique_frames)

        files_to_preserve = {frame.image_name for frame in unique_frames}
        all_files = {f for f in os.listdir(imgs_dir_path)}

        files_to_remove = all_files - files_to_preserve
        
        for f in sorted(files_to_remove):
            os.remove(os.path.join(imgs_dir_path, f))
            logging.info(f'Removed file sequence for camera: {f}')


def main() -> None:
    parser = argparse.ArgumentParser(
                    prog='RemoveSimilarFrames',
                    description='The utility to remove similar frames from the directory with images. All files, which are non-image files, corrupted image files, or image files with small images (see --min_img_size_to_process param) WILL BE REMOVED')
    
    parser.add_argument('imgs_dir_path', help='path to the direcory with images to process. The images names should be in the following formats: {camera id}-{epoch timestamp ms}.{image file extension} or {camera id}__YYYY_MM_DD__HH_MM_SS.{image file extension} with Y, M, D, H, M, S stand for corresponding digits of year, month, day, hour, minute and second. images for different cameras are assumed to be different, no cross comparison is performed')
    parser.add_argument('--cache_size', type=int, help='size of the images cache. By defaut, the utility loads all the images to speed up the process')
    parser.add_argument('--min_img_size_to_process', type=int, default=300, help='minimal size of the side for the image to be processed. Images with smaller sides are not being processed and will be removed')
    parser.add_argument('--gb_half_ksize_ratio', type=float, default=0.0075, help='this parameter adjusts the size of the gaussian blur kernel. The value is equal to half of the kernel size divided by the smallest side of the smallest area image among images to process')
    parser.add_argument('--min_cnt_area_ratio', type=float, default=0.0125, help='this parameter adjusts the minimal area of the images difference to process in image comparison. The value is equal to minimal area of the contour divided by the smallest area of image among images to process')
    parser.add_argument('--similarity_threshold', type=float, default=0.01, help='this parameter adjusts the threshold of minimal area of the images difference for two images to be considered non-similar. The value is equal to minimal area of the contour divided by the smallest area of image among images to process. If the metric score is lower than this value, images are considered similar. This parameter must have smaller value than --min_cnt_area_ratio param')
    parser.add_argument('--verbose', type=bool, default=True, help='this paramater enables verbose printing of the algorithm stages to the console')

    args = parser.parse_args()

    assert args.similarity_threshold < args.min_cnt_area_ratio, 'Please adjust the value of similarity_threshold to be lower than min_cnt_area_ratio'

    logging.basicConfig(level = logging.INFO if args.verbose else logging.ERROR)

    remover = SimilarImagesRemover(args.gb_half_ksize_ratio, args.min_cnt_area_ratio, 
                                   args.similarity_threshold, args.cache_size, args.min_img_size_to_process)
    

    remover.process_directory(args.imgs_dir_path)



if __name__ == '__main__':
    main()