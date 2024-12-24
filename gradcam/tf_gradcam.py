import tensorflow as tf
import keras
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model 

class TensorflowGradCAM:
    def __init__(self):
        self.model=ResNet50(weights='imagenet')
        self.layer_name='conv5_block3_out'
    
    def generate_heatmap(self,img_path,target_class=None):

        img=image.load_img(img_path,target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = tf.expand_dims(img_array, axis=0)
        
        # Create gradient model
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        
        # Generate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if target_class is None:
                target_class = tf.argmax(predictions[0])
            loss = predictions[:, target_class]
            
        # Extract gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Generate heatmap
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs), axis=-1
        )
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy().squeeze(), img    
