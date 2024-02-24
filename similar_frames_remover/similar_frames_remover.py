import os
from typing import List, Callable, Dict, Union, Tuple
import numpy as np
from thirdrdparty.imaging_interview import compare_frames_change_detection
from itertools import compress
from sklearn.cluster import DBSCAN
import logging
from similar_frames_remover.annotation import FrameAnnotation, create_annotation
from similar_frames_remover.img_processing import create_caching_load_preprocess, get_min_area_img_size


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


def remove_subsequent_similar_frames(img_seq: List[FrameAnnotation], dataset_path: str, resize_img_size_wh: Tuple[int, int],
                                     load_imgs_function: Callable[[str, str, Tuple[int, int], Union[float, None]], np.ndarray], 
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
                                        dataset_path: str, resize_img_size_wh: Tuple[int, int],
                                        load_imgs_function: Callable[[str, str, Tuple[int, int], 
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


class SimilarFramesRemover:
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