import argparse
import logging
import os
from pathlib import Path


FULL_DATASET_FILE = 'datasets/full_dataset.txt'
SHORT_DATASET_FILE = 'datasets/short_dataset.txt'


def parse():
    """CLI parser."""
    parser = argparse.ArgumentParser('Parser for creation of the full benchmark dataset list.')
    parser.add_argument('--count', type=int, default=8, help='Count of gt images per kernel.')
    return parser.parse_args()


def generate_full_list():
    args = parse()

    mult_const = 10
    gt_images = {
        'sun': sorted(list(Path('datasets/gt/Sun-gray').rglob('*.png'))) * mult_const,
        'sca2023-animals': sorted(list(Path('datasets/gt/precomp/animals').rglob('*.jpg'))) * mult_const,
        'sca2023-city': sorted(list(Path('datasets/gt/precomp/city').rglob('*.jpg'))) * mult_const,
        'sca2023-faces': sorted(list(Path('datasets/gt/precomp/faces').rglob('*.jpg'))) * mult_const,
        'sca2023-icons': sorted(list(Path('datasets/gt/precomp/icons_jpg').rglob('*.png'))) * mult_const,
        'sca2023-nature': sorted(list(Path('datasets/gt/precomp/nature').rglob('*.jpg'))) * mult_const,
        'sca2023-texts': sorted(list(Path('datasets/gt/precomp/texts').rglob('*.jpg'))) * mult_const,
    }

    for dataset_type, files in gt_images.items():
        logging.info(f'Dataset {dataset_type} has {len(files) // mult_const} files.')

    num_datasets = len(gt_images.keys())

    kernels = {
        'eye_blur': list(Path('datasets/kernels/eye-psf/processed/synthetic').rglob('*.npy')),
        'gauss_blur': list(Path('datasets/kernels/gauss-blur/processed/synthetic').rglob('*.npy')),
        'motion_blur': list(Path('datasets/kernels/motion-blur/processed').rglob('*.npy')),
    }

    for blur_type, files in kernels.items():
        logging.info(f'Blur type {blur_type} has {len(files)} files.')

    with open(FULL_DATASET_FILE, 'w+') as file:
        file.write('blur_type, blur_dataset, kernel, image_dataset, image\n')
        for blur_type in kernels.keys():
            for kernel in kernels[blur_type]:
                blur_dataset = str(kernel).split(os.sep)[-2]
                for dataset_type in gt_images.keys():
                    for _ in range(args.count // num_datasets):
                        image = gt_images[dataset_type].pop(0)
                        file.write(f'{blur_type},{blur_dataset},{kernel},{dataset_type},{image}\n')
    
    logging.info(f'Full benchmark dataset list was generated. Path: {FULL_DATASET_FILE}')


def generate_short_list():
    with open(FULL_DATASET_FILE, 'r+') as full_list, open(SHORT_DATASET_FILE, 'w+') as short_list:
        for i, line in enumerate(full_list):
            if i % 10 == 0:
                short_list.write(line)

    logging.info(f'Short benchmark dataset list was generated. Path: {SHORT_DATASET_FILE}')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate_full_list()
    generate_short_list()
