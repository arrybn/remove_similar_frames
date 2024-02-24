import argparse
import logging
from similar_frames_remover.similar_frames_remover import SimilarFramesRemover

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

    remover = SimilarFramesRemover(args.gb_half_ksize_ratio, args.min_cnt_area_ratio, 
                                   args.similarity_threshold, args.cache_size, args.min_img_size_to_process)
    

    remover.process_directory(args.imgs_dir_path)


if __name__ == '__main__':
    main()