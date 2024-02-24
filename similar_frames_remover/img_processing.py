import copy
import os
from functools import lru_cache
from typing import Callable, Dict, List, Tuple, Union

import cv2
import numpy as np

from similar_frames_remover.annotation import FrameAnnotation
from thirdrdparty.imaging_interview import preprocess_image_change_detection


def get_gaussian_blur_ksize(
    img_size_wh: Tuple[int, int], gaussian_blur_ksize_ratio: Union[float, None]
):
    gaussian_blur_ksize = None
    if gaussian_blur_ksize_ratio is not None and gaussian_blur_ksize_ratio > 0.0:
        gaussian_blur_ksize = [
            max(2 * int(min(img_size_wh) * gaussian_blur_ksize_ratio) + 1, 3)
        ]

    return gaussian_blur_ksize


def preprocess_imgs(
    imgs: List[np.ndarray], gblur_half_size_ratio: Union[float, None] = 0.025
) -> List[np.ndarray]:
    imgs_shapes = np.asarray([i.shape[:-1] for i in imgs])

    preprocessed_imgs = []

    if not (imgs_shapes == imgs_shapes[0]).all():
        areas = imgs_shapes[:, 0] * imgs_shapes[:, 1]
        shape_to_resize = imgs_shapes[np.argmin(areas)]
        preprocessed_imgs = [
            (
                cv2.resize(i, shape_to_resize[::-1])
                if not (i.shape[:-1] == shape_to_resize).all()
                else i
            )
            for i in imgs
        ]
    else:
        preprocessed_imgs = copy.deepcopy(imgs)

    gaussian_blur_ksize = get_gaussian_blur_ksize(imgs_shapes[0], gblur_half_size_ratio)
    # if gblur_half_size_ratio is not None and gblur_half_size_ratio > 0.0:
    # gaussian_blur_ksize = [max(2*int(.min()*gblur_half_size_ratio) + 1, 3)]

    preprocessed_imgs = [
        preprocess_image_change_detection(i, gaussian_blur_ksize)
        for i in preprocessed_imgs
    ]

    return preprocessed_imgs


def create_caching_load_preprocess(
    cache_size=None,
) -> Callable[[str, str, tuple[int, int], Union[float, None]], np.ndarray]:
    @lru_cache(maxsize=cache_size)
    def load_preprocess(
        img_name: str,
        dataset_path: str,
        out_img_size_wh: tuple[int, int],
        gaussian_blur_ksize_ratio: Union[float, None],
    ) -> np.ndarray:
        img = cv2.imread(os.path.join(dataset_path, img_name), cv2.IMREAD_COLOR)

        img = cv2.resize(img, out_img_size_wh)
        gaussian_blur_ksize = get_gaussian_blur_ksize(
            out_img_size_wh, gaussian_blur_ksize_ratio
        )

        return preprocess_image_change_detection(img, gaussian_blur_ksize)

    return load_preprocess


def get_min_area_img_size(img_seq: List[FrameAnnotation]) -> tuple[int, int]:
    min_area_img_sample = min(
        img_seq, key=lambda sample: np.prod(sample.orig_img_size_wh)
    )
    out_img_size_wh = min_area_img_sample.orig_img_size_wh

    return out_img_size_wh
