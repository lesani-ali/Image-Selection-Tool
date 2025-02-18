from typing import List


class Base(object):
    def __init__(self, data_path: str, filenames: List[str]):
        self.data_path = data_path
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

