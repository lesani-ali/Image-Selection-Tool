from tqdm import tqdm
import datasets.Base
from models.object_detection import ObjectDetectionModel
import matplotlib.pyplot as plt
import datasets
from util.utils import write_filename
import logging
import cv2


def run_inference(
    model: ObjectDetectionModel,
    dataset: datasets.Base,
    output_dir: str,
) -> None:

    num_images = len(dataset)

    for idx in tqdm(range(num_images), desc="Processing images", ncols=100):
        # Get inputs
        inputs = dataset[idx]
        if not inputs:
            continue

        folder_name = inputs['folder_name']
        file_name = inputs['file_name']

        logging.info(
            f"\nProcessing image {idx + 1}/{num_images}: {folder_name}{file_name}"
        )

        # Process input image
        boxes = model(inputs['image'])

        if len(boxes) == 0:
            logging.info('Desired object is not detected.')
            continue

        # Add bounding boxes to the image
        for b in boxes:
            cv2.rectangle(inputs['image'], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 3)

        # Save the output image
        dataset.save_outputs(output_dir, inputs)

        # Save the filename
        write_filename(
            output_dir, f"{folder_name}{file_name}"
        )

        logging.info(
            f"Output image saved at: {output_dir}/{folder_name}"
        )
