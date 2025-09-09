import json
import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def _render_sketch(vector_image, side: int = 256, time_frac: float = None, skip_front: float = False) -> np.ndarray:
    """Render a raster image from raw data stored in json.
    """
    raster_image = np.ones((side, side), dtype=np.float32)
    prevX, prevY = None, None
    start_time = vector_image[0]['timestamp']
    end_time = vector_image[-1]['timestamp']
    full_time = end_time - start_time

    # Skip drawing a percentage of the first or last strokes of the sketch.
    if time_frac:
        if skip_front:
            start_time += full_time * time_frac
        else:
            end_time -= full_time * time_frac

    for points in vector_image:
        time = start_time
        if time > end_time:
            break

        x, y = map(float, points['coordinates'])
        x = int(x * side)
        y = int(y * side)
        pen_state = list(map(int, points['pen_state']))
        if not (prevX is None or prevY is None):
            # Draw a line using OpenCV
            if 0 <= prevX < side and 0 <= prevY < side and 0 <= x < side and 0 <= y < side:
                cv2.line(raster_image, (prevX, prevY), (x, y), color=0, thickness=1)
            if pen_state == [0, 1, 0]:
                prevX = x
                prevY = y
            elif pen_state == [1, 0, 0]:
                prevX = None
                prevY = None
            else:
                raise ValueError('pen_state not accounted for')
        else:
            prevX = x
            prevY = y
    # invert black and white pixels and dilate
    raster_image = (1 - cv2.dilate(1 - raster_image, np.ones((3, 3), np.uint8), iterations=1)) * 255
    return raster_image.astype(np.uint8)


def _render_fscoco(root: str):
    """Render all sketches in FSCOCO dataset."""
    for i in range(1, 101):
        files = os.listdir(os.path.join(root, f"raw_data/{i}"))
        for file in files:
            file = file[:-5]
            with open(os.path.join(root, f"raw_data/{i}/{file}.json"), encoding="utf-8") as json_file:
                data = json.load(json_file)
            sketch = _render_sketch(data)
            png = Image.fromarray(sketch)
            png.save(os.path.join(root, f"raster_sketches/{i}/{file}.png"), format='PNG')


class SampleCollection:

    def __init__(self, samples=None):
        self._samples = [] if samples is None else samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        sample = self._get_sample(idx)
        return self._get_image(sample), self._get_sketch(sample)

    def __bool__(self):
        return len(self) > 0

    def _get_sample(self, idx):
        if isinstance(idx, int):
            return self._samples[idx]
        else:
            raise TypeError("Index must be an integer")

    def _train_split(self):
        return [i for i in range(len(self)) if i % 3 == 0]

    def _test_split(self):
        return [i for i in range(len(self)) if i % 3 == 1]

    def _validate_split(self):
        return [i for i in range(len(self)) if i % 3 == 2]

    def split(self):
        train = SampleCollectionSubset(self, self._train_split())
        test = SampleCollectionSubset(self, self._test_split())
        validate = SampleCollectionSubset(self, self._validate_split())
        return train, test, validate

    def _get_image(self, sample):
        raise NotImplementedError("This method should be implemented in subclasses")

    def _get_sketch(self, sample):
        raise NotImplementedError("This method should be implemented in subclasses")

    def name(self):
        raise NotImplementedError("This method should be implemented in subclasses")


class SampleCollectionSubset(SampleCollection):

    def __init__(self, collection: SampleCollection, indices: list):
        super().__init__([collection._get_sample(i) for i in indices])
        self._collection = collection

    def _get_image(self, sample):
        return self._collection._get_image(sample)

    def _get_sketch(self, sample):
        return self._collection._get_sketch(sample)

    def name(self):
        return self._collection.name()
    
    def file_names(self):
        return self._samples


class DatasetFSCOCO(SampleCollection):

    def __init__(self, root, unseen=True):

        samples = []

        for i in range(1, 101):
            file_names = os.listdir(os.path.join(root, "images", str(i)))
            file_names = [file.split('.')[0] for file in file_names]
            # Sort the file names
            file_names.sort()

            for file_name in file_names:
                samples.append((i, file_name))

        super().__init__(samples)
        self.root = root

        # Test if sketches are already rendered as PNG
        sketch_path = self._get_sketch(self._get_sample(0))
        if not os.path.exists(sketch_path):
            print("Rendering sketches, please wait ...")
            os.makedirs(os.path.join(root, "raster_sketches"), exist_ok=True)
            _render_fscoco(root)

        val_path = "val_unseen_user.txt" if unseen else "val_normal.txt"
        with open(os.path.join(self.root, val_path), 'r') as f:
            lines = f.readlines()
            self.val_ids = set(map(int, lines))

    def _get_image(self, sample):
        dir, file = sample
        file_path = os.path.join(self.root, "images", str(dir), file + ".jpg")
        return file_path

    def _get_sketch(self, sample):
        dir, file = sample
        file_path = os.path.join(self.root, "raster_sketches", str(dir), file + ".png")
        return file_path

    def _test_split(self):
        return [i for i, (_, idx) in enumerate(self._samples) if int(idx) in self.val_ids]

    def _train_split(self):
        return [i for i, (_, idx) in enumerate(self._samples) if int(idx) not in self.val_ids]

    def _validate_split(self):
        return []

    def name(self):
        return "fscoco"


class DatasetSketchyCOCO(SampleCollection):
    def __init__(self, root):
        train_files = os.listdir(os.path.join(self.root, "train", "image"))
        test_files = os.listdir(os.path.join(self.root, "test", "image"))

        train_files = [("train", file) for file in train_files]
        test_files = [("test", file) for file in test_files]

        super().__init__(train_files + test_files)
        self._train_indices = [i for i in range(len(train_files))]
        self._test_indices = [i + len(train_files) for i in range(len(test_files))]
        self._root = root

    def _train_split(self):
        return self._train_indices

    def _test_split(self):
        return self._test_indices

    def _validate_split(self):
        return []

    def _get_image(self, sample):
        split, file = sample
        return os.path.join(self._root, split, "image", file)

    def _get_sketch(self, sample):
        split, file = sample
        file = file[:file.rindex("-")] + ".png"
        return os.path.join(self._root, split, "sketch", file)

    def name(self):
        return "sketchycoco"


class DatasetAdapter(Dataset):
    def __init__(self, collection, transforms_sketch=None, transforms_image=None):
        self.collection = collection
        self.transforms_sketch = transforms_sketch
        self.transforms_image = transforms_image

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, idx):
        image, sketch = self.collection[idx]

        if isinstance(sketch, str):
            sketch = Image.open(sketch).convert('L')
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        if self.transforms_sketch:
            sketch = self.transforms_sketch(sketch)

        if self.transforms_image:
            image = self.transforms_image(image)

        # We are returning the index as well to track which sample it is for loss purposes
        return sketch, image, idx

    def name(self):
        return self.collection.name()


def load_dataset(root, mode=None) -> SampleCollection:
    """
    Identify the dataset type based on the directory structure.
    Args:
        root (str): The root directory of the dataset.
        mode (str, optional): The mode of the dataset, in case of FSCoco dataset, it can be "normal" or "unseen".
    Returns:
        SampleCollection: The appropriate dataset class based on the structure.
    """

    root = os.path.abspath(root)

    if os.path.exists(os.path.join(root, "raw_data")):
        assert mode in ["normal", "unseen"]
        return DatasetFSCOCO(root, unseen=(mode == "unseen"))
    elif os.path.exists(os.path.join(root, "sketchy", "train", "image")):
        return DatasetSketchyCOCO(root)
    else:
        raise ValueError(f"Unknown dataset structure in {root}. Please provide a valid dataset directory.")
