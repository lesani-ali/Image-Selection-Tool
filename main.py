import sys
import os
sys.path.insert(0, './')
current_dir = os.path.dirname(os.path.abspath(__file__))

import argparse
import logging
import warnings
from types import SimpleNamespace
from models.object_detection import ObjectDetectionModel
from util.utils import setup_logger, load_config, readlines
from util.inference import run_inference
import datasets

warnings.filterwarnings("ignore")  # Suppress all warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run image selection tool.'
    )
    parser.add_argument(
        '--config-dir', type=str,
        default=f'{current_dir}/config/config.yaml',
        help='Path to the configuration file.'
    )
    parser.add_argument(
        '--input-dir', type=str,
        default=f'{current_dir}/data/inputs',
        help='Path to the input data.'
    )
    parser.add_argument(
        '--dataset', type=str,
        default='MyData',
        help='Name of dataset to be used.'
    )
    parser.add_argument(
        '--data-filename', type=str,
        default=f'{current_dir}/data/inputs/paths.txt',
        help='Path to the input data.'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default=f'{current_dir}/data/outputs',
        help='Path to save the output data.'
    )
    parser.add_argument(
        '--objects', type=str,
        default=None,
        help='Objects to be detected.'
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config_dir)
    config = SimpleNamespace(**config)

    if args.objects is not None:
        config.object_detection['text_prompt'] = args.objects

    setup_logger(f'{current_dir}/logs')

    logging.info('Starting image selection tool...')
    logging.info(f'Configuration file: {args.config_dir}')
    logging.info(f'Input directory: {args.input_dir}')
    logging.info(f'Output directory: {args.output_dir}')

    logging.info('\nInstantiating the object detection model...')
    model = ObjectDetectionModel(config.object_detection)

    file_path = os.path.join(args.input_dir, "{}".format(args.data_filename))
    filenames = readlines(file_path)

    class_name = f"{args.dataset}"
    data_cls = getattr(datasets, class_name, None)
    dataset = data_cls(args.input_dir, filenames)

    run_inference(
        model,
        dataset,
        args.output_dir,
    )

    logging.info('\n Desired images selected successfully.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
