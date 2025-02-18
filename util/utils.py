import logging
import sys
import os
import yaml


def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def write_filename(output_dir: str, filename: str) -> None:
    with open(os.path.join(output_dir, 'filenames.txt'), 'a') as f:
        f.write(f'{filename}\n')


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def setup_logger(log_path: str, log_to_stdout: bool = False) -> None:
    logger = logging.getLogger()

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    if log_to_stdout:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handler = logging.FileHandler(f'{log_path}/log_file.log', 'w', 'utf-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

