from utils import common
from selector.onnx.detector import NerveSegmenter
from selector.extractor import FeatureExtractor

import cv2
import numpy as np
import os
import torch
import random
from torchvision import models, transforms
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from PIL.Image import open, fromarray
from tqdm import tqdm

from typing import Sequence
from collections import defaultdict

random.seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_dir = os.path.join(os.path.dirname(__file__), 'models')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


class Image:
    _index = 0

    def __init__(self, path):
        # auto index
        self.index = Image._index
        Image._index += 1

        self.path = path
        self.raw_image = common.load_image(path)  # for show
        self.image = open(path).convert('RGB')  # for compute
        self.feature = None


class ImageBag:
    def __init__(self, images: Sequence[Image]):
        self.images = {image.index: image for image in images}
        self.models = {}
        self.clusters = {}

    def from_dir(self, path):
        images = (Image(os.path.join(path, x)) for x in os.listdir(path))
        self.images = {image.index: image for image in images}

    def load_models(self, alpha=1.0):
        segmenter = NerveSegmenter(os.path.join(model_dir, 'nerve.onnx'))

        quality_filter = models.resnet18(pretrained=False)
        num_ftrs = quality_filter.fc.in_features
        quality_filter.fc = torch.nn.Linear(num_ftrs, 2)  # Assuming two classes
        quality_filter.load_state_dict(torch.load(os.path.join(model_dir, 'quality.pth'), map_location=device))
        quality_filter.to(device)
        quality_filter.eval()

        # classifier = models.resnet50(pretrained=True)
        # classifier = torch.nn.Sequential(*list(classifier.children())[:-1])
        classifier = FeatureExtractor(alpha=alpha)
        classifier.to(device)
        classifier.eval()

        self.models['segmenter'] = segmenter
        self.models['filter'] = quality_filter
        self.models['classifier'] = classifier

    def qc(self, inplace=True, threshold=.9):
        good_indexes = []
        for index, image in tqdm(self.images.items()):
            input_tensor = transform(image.image)
            input_batch = input_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                output = self.models['filter'](input_batch)
                pred = torch.nn.functional.softmax(output)[0][1].cpu().numpy()
                class_idx = 1 if pred >= threshold else 0
            if class_idx == 1:
                good_indexes.append(index)

        if inplace:
            self.images = {k: v for k, v in self.images.items() if k in good_indexes}

        return good_indexes

    def classify(self, n_clusters=8):
        features, indexes = [], []
        for index, image in tqdm(self.images.items()):
            binary = self.models['segmenter'](np.array(image.image))
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            input_tensor = transform(image.image), transform(fromarray(binary))
            input_batch = (it.unsqueeze(0).to(device) for it in input_tensor)
            with torch.no_grad():
                output = self.models['classifier'](*input_batch)
                output = output.view(output.size(0), -1)
                feature = output.cpu().numpy()

                image.feature = feature
                features.append(feature)
                indexes.append(index)

        features = np.vstack(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.models['kmeans'] = kmeans
        labels = kmeans.fit_predict(features)

        self.clusters = defaultdict(list)
        for idx, label in zip(indexes, labels):
            self.clusters[label].append(self.images[idx])

        return labels

    def choose(self, n_center=1, inplace=True):
        representative_images = []
        kmeans = self.models['kmeans']
        for cluster_idx, images in self.clusters.items():
            features = np.vstack([i.feature for i in images])
            cluster_center = kmeans.cluster_centers_[cluster_idx].reshape(1, -1)
            distances = cdist(features, cluster_center, metric='euclidean').flatten()
            closest_image_indices = np.argsort(distances)[:n_center]
            representative_images.append((cluster_idx, closest_image_indices))

        if inplace:
            images_dict = {}
            for cluster_idx, idxes in representative_images:
                for i, idx in enumerate(idxes):
                    image = self.clusters[cluster_idx][idx]
                    images_dict[image.index] = image
            self.images = images_dict

        return representative_images


if __name__ == '__main__':
    print(model_dir)
