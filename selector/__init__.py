import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Global config:
max_num_keypoints = 2048
# ['superpoint', 'disk', 'aliked', 'sift', 'doghardnet']
extractor = 'superpoint'
