import os
import random
from typing import Any

import numpy as np
import scipy.spatial.distance as ssd
import torch
from typing_extensions import Sequence


# A context manager proxy for any object
class cmp:
    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self.obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __getattr__(self, name):
        return getattr(self.obj, name)

    def __setattr__(self, name, value):
        if name == 'obj':
            self.__dict__[name] = value
        else:
            setattr(self.obj, name, value)

    def __call__(self, *args, **kwargs):
        return self.obj(*args, **kwargs)


def seed_everything(seed: int):
    """Seed random generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_distances(sketch_feats, image_feats):
    return ssd.cdist(sketch_feats, image_feats, 'sqeuclidean')


def _output_html(sketch_index: int, image_indices: Sequence[int], file_names) -> str:
    """Generate an HTML visualizing the results on the evaluated dataset."""
    tmp_line = "<tr>"
    tmp_line += "<td><image src='%s' width=256 /></td>" % (
        os.path.join("fscoco", "raster_sketches", str(file_names[sketch_index][0]), str(file_names[sketch_index][1]) + ".png"))
    for i in image_indices:
        if i != sketch_index:
            tmp_line += "<td><image src='%s' width=256 /></td>" % (
                os.path.join("fscoco", "images", str(file_names[i][0]), str(file_names[i][1]) + ".jpg"))
        else:
            tmp_line += "<td ><image src='%s' width=256   style='border:solid 2px red' /></td>" % (
                os.path.join("fscoco", "images", str(file_names[i][0]), str(file_names[i][1]) + ".jpg"))

    return tmp_line + "</tr>"


def calculate_results(dist: object, file_names: object = None, labels: Sequence[Any] = None) \
        -> tuple[int, int, int, int]:
    """Calculate results of the evalutation

    If file_names is not None will also create an HTML visualization of the results.
    (Only supports FSCOCO dataset)
    """
    if labels is None:
        labels = np.arange(len(dist))
    top1 = 0
    top5 = 0
    top10 = 0
    tmp_line = ""
    for i in range(dist.shape[0]):
        rank = dist[i].argsort()
        if rank[0] == labels[i]:
            top1 = top1 + 1
        if labels[i] in rank[:5]:
            top5 = top5 + 1
        if labels[i] in rank[:10]:
            top10 = top10 + 1
        if file_names is not None:
            tmp_line += _output_html(i, rank[:10], file_names) + "\n"
    num = dist.shape[0]

    if file_names is not None:
        html_content = """
        <html>
        <head></head>
        <body>
        <table>%s</table>
        </body>
        </html>""" % tmp_line
        with open(r"result.html", 'w+') as f:
            f.write(html_content)
    return top1, top5, top10, num


def expand_file_names(file_names: str | list[str]) -> list[str]:
    """
    Expand the given file patterns to full file paths.
    Args:
        file_names (list): List of file names or patterns.
    """
    import os
    import glob

    # If not list, convert to list
    if not isinstance(file_names, list):
        file_names = [file_names]

    expanded_file_names = []
    for file_name in file_names:
        if os.path.isfile(file_name):
            expanded_file_names.append(file_name)
        elif os.path.isdir(file_name):
            # If the file name is a directory, expand it to all files in the directory
            expanded_file_names.extend([os.path.join(file_name, f) for f in os.listdir(file_name)])
        else:
            # If the file name is a pattern, expand it using glob
            expanded_file_names.extend(glob.glob(file_name))

    return expanded_file_names
