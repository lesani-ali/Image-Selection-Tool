import cv2
import numpy as np
from typing import List, Optional
import os
from PIL import Image
from .Base import Base


class MyData(Base):
    def __init__(self, data_path: str, filenames: List[str]):
        super(MyData, self).__init__(data_path, filenames)

    def __getitem__(self, index: int) -> Optional[np.ndarray]:
        line = self.filenames[index]
        path = line

        img_path = os.path.join(self.data_path, path + '.png')
        gt_path = os.path.join(self.data_path, path + '_depth.npy')

        inputs = {
            'image': self.load_image(img_path),
            'ground_truth': self.load_depth(gt_path),
            'folder_name': path[:-5],
            'file_name': path[-5:],
        }

        return inputs

    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_depth(self, depth_path: str) -> np.ndarray:
        return np.load(depth_path)

    def save_outputs(self, output_dir: str, outputs) -> None:
        out_dir = os.path.join(output_dir, outputs['folder_name'])
        os.makedirs(out_dir, exist_ok=True)

        # Save input image
        img = Image.fromarray(outputs['image'])
        img.save(os.path.join(out_dir, outputs['file_name'] + '.png'))

        # Save ground truth depth
        gt_depth = outputs['ground_truth']
        np.save(
            os.path.join(out_dir, outputs['file_name'] + '_depth.npy'), gt_depth
            )

