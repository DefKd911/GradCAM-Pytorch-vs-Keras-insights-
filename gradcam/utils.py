import cv2
import numpy as np
from PIL import Image

def visualize_gradcam(heatmap,original_img,alpha=0.4):
    if isinstance(original_img,Image.Image):
        original_img=np.array(original_img)
    
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    heatmap_rgb=np.uint8(255*heatmap_resized)
    heatmap_rgb=cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(original_img,
                                   1-alpha,
                                   heatmap_rgb,
                                   alpha,
                                   0)
    return superimposed


