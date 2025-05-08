
# img_bgr = cv2.imread("/media/huifang/data/fiducial/annotations/overlap_annotation/103.png")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 1. Load & convert the image
img_bgr = cv2.imread("/media/huifang/data/fiducial/annotations/overlap_annotation/119.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

# 2. Prepare pixel data for GMM
pixels = img_rgb.reshape(-1, 3).astype(np.float64)

# 3. Fit a 3-component Gaussian Mixture Model in RGB space
gmm = GaussianMixture(n_components=3, covariance_type='tied', random_state=0)
gmm.fit(pixels)
labels = gmm.predict(pixels).reshape(h, w)

# 4. Define overlay colors for each cluster (R, G, B)
overlay_colors = {
    0: np.array([255,   0,   0]),   # red
    1: np.array([  0, 255,   0]),   # green
    2: np.array([  0,   0, 255]),   # blue
}

# 5. Blend the cluster colors onto the original image
alpha = 0.5  # transparency factor
overlay = img_rgb.astype(np.float64).copy()
for cls, color in overlay_colors.items():
    mask = (labels == cls)
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

overlay = np.clip(overlay, 0, 255).astype(np.uint8)

# 6. Display original and overlay side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img_rgb)
ax1.set_title("Original Image")
ax1.axis("off")

ax2.imshow(overlay)
ax2.set_title("GMM (3-component) Overlay")
ax2.axis("off")

plt.tight_layout()
plt.show()

