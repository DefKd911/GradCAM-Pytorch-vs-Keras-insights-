import matplotlib.pyplot as plt
import os
from gradcam.pytorch_gradcam import PytorchGradCAM
from gradcam.tf_gradcam import TensorflowGradCAM
from gradcam.utils import visualize_gradcam


def process_multiple_images(image_paths):
    pytorch_gradcam = PytorchGradCAM()
    tensorflow_gradcam = TensorflowGradCAM()
    
    for img_path in image_paths:
        # Generate heatmaps
        pytorch_heatmap, pytorch_img = pytorch_gradcam.generate_heatmap(img_path)
        tensorflow_heatmap, tf_img = tensorflow_gradcam.generate_heatmap(img_path)
                                                                    
        # Create plot for current image
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Analysis for {os.path.basename(img_path)}')
        
        # Plot original images and heatmaps
        axes[0,0].imshow(pytorch_img)
        axes[0,0].set_title('Original Image (PyTorch)')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(tf_img)
        axes[0,1].set_title('Original Image (TensorFlow)')
        axes[0,1].axis('off')
        
        pytorch_viz = visualize_gradcam(pytorch_heatmap, pytorch_img)
        tensorflow_viz = visualize_gradcam(tensorflow_heatmap, tf_img)
        
        axes[1,0].imshow(pytorch_viz)
        axes[1,0].set_title('GradCAM (PyTorch)')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(tensorflow_viz)
        axes[1,1].set_title('GradCAM (TensorFlow)')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    image_paths = [
        "test_images\pexels-photo-326012.jpeg",
        "test_images\download.jpg",
        "test_images\squirrel-animal-cute-rodents-47547.jpeg"
    ]
    process_multiple_images(image_paths)