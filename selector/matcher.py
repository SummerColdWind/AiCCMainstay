from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from selector import device, max_num_keypoints, extractor
from itertools import combinations
from tqdm import tqdm

class Matcher:
    def __init__(self):
        self.superpoint = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
        self.lightglue = LightGlue(features='superpoint').eval().to(device)
        self.table = {}

    def match_similarity(self, bag):
        self.table = {}
        combo = combinations(bag.images.keys(), 2)
        for x, y in tqdm(list(combo)):
            image_1, image_2 = bag.images[x], bag.images[y]
            feats0, feats1 = (
                self.superpoint.extract(load_image(image_1.path).to(device)),
                self.superpoint.extract(load_image(image_2.path).to(device))
            )
            matches01 = self.lightglue({'image0': feats0, 'image1': feats1})

            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

            kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
            similarity = matches.shape[0] / (kpts0.shape[0] + kpts1.shape[0]) * 2
            self.table[(image_1.index, image_2.index)] = similarity

        if self.table:
            return max(self.table.values())
        return 0
