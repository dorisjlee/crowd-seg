# runs ../segment/segment.cpp to segment images by color

import glob
import os

BASE_DIR = '/Users/akashds1/Dropbox/CrowdSourcing/Image-Segmentation/'
OUR_DIR = BASE_DIR + 'Vision-stuff/'
INPUT_IMG_DIR = OUR_DIR + 'COCO-ppm/'
OUTPUT_IMG_DIR = OUR_DIR + 'color-segmented-images/'
CODE_DIR = BASE_DIR + 'segment/'


def compile_code():
    os.system('cd {}; make'.format(CODE_DIR))


def segment_image_by_color(
    sigma, k, m,
    input_img_path, output_img_path
):
    os.system(
        'cd {}; ./segment {} {} {} {} {}'.format(
            CODE_DIR,
            sigma, k, m,
            input_img_path, output_img_path
        )
    )


def segment_all():
    compile_code()
    for input_img_path in glob.glob('{}*'.format(INPUT_IMG_DIR)):
        img_name = input_img_path.split('/')[-1].split('.')[0]
        output_img_path = '{}{}.png'.format(OUTPUT_IMG_DIR, img_name)
        # print output_img_path
        if img_name:
            # currently using same params for all images
            sigma = 0.5
            k = 500
            m = 20
        segment_image_by_color(sigma, k, m, input_img_path, output_img_path)


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)
    segment_all()
