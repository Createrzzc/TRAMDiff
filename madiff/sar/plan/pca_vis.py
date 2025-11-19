import torch
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image

def pca_vis(feature_in, save_path="reduced_features.png"):
    features_2d = features_in.detach().numpy()
    pca = PCA(n_components=3)
    features_reduced = pca.fit_transform(features_2d)
    reduced_image = (features_reduced - features_reduced.min(axis=0)) / (features_reduced.max(axis=0) - features_reduced.min(axis=0)) * 255
    reduced_image = reduced_image.reshape(h, w, 3).astype(np.uint8)  # shape: (h, w, 3)
    image = Image.fromarray(reduced_image)
    image.save(save_path)