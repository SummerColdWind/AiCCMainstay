# AiCCMainstay

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> This is the **Automated Program Using Convolutional Neural Networks for Objective and Reproducible Selection of Corneal Confocal Microscopy Images** code repository.


## About

AiCCMainstay: An objective and repeatable image selection algorithm for corneal confocal microscopy based on deep learning

AiCCMainstay is an open-source project designed to select a specified number of the most representative portions from a large number of corneal confocal microscope raw images.It is licensed under the GNU
General Public License v3.0 (GPL-3.0).

## Cite our paper

```
Qiao Q, Xue W, Li J, et al. Automated program using convolutional neural networks for objective and reproducible selection of corneal confocal microscopy images. DIGITAL HEALTH. 2025;11. doi:10.1177/20552076251326223`
```

## Features
<hr>

- Fast and efficient
- Without subjective preference
- Open source


## How to use
<hr>

To use this project, follow these steps:

1. Environmental preparation

This is our conda environment, which you can reproduce with the following command:
```shell
conda env create -f environment.yml
```
More concise environment requirements will be provided in the future. `:)`

2. Example
```python
# Import the necessary dependencies
from selector.workflow import ImageBag
from utils.common import save_image
import os

# Instantiate an empty ImageBag object
bag = ImageBag([])
# To read all images from a directory, you must ensure that the directory is full of image files
bag.from_dir(r'path/to/your/image/dir')
# Load the deep learning model.
# The alpha parameter specifies the ratio of feature fusion, which is recommended to be 1.0
bag.load_models(alpha=1.0)
# Filter out low quality images. threshold specifies the threshold for the quality score.
# If you find that fewer images are filtered, adjust this parameter as appropriate.
bag.qc(threshold=0.9)
# Cluster.
# n_clusters specifies the number of clusters, defined according to your needs
bag.classify(n_clusters=4)
# Select representative images from the clusters of each cluster.
# n_center specifies the number of selections, again defined according to your needs
rst = bag.choose(n_center=2)


# Write out the image of each cluster
for cluster_idx, images in bag.clusters.items():
    save_dir = f'rst/{cluster_idx}'
    os.makedirs(save_dir, exist_ok=True)
    for idx, image in enumerate(images):
        save_image(image.raw_image, os.path.join(save_dir, f'{idx}.png'))

# Write out all the representative images
for cluster_idx, idxes in rst:
    for i, idx in enumerate(idxes):
        save_image(bag.clusters[cluster_idx][idx].raw_image, f'rst/{cluster_idx}_{i}_rp.png')
```
