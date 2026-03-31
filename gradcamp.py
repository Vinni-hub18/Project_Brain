import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
def get_gradcam_heatmap(model, img_array, last_conv_layer_name):

    # ✅ Safe layer fetch
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except:
        raise ValueError(f"❌ Layer '{last_conv_layer_name}' not found")

    # ✅ FIX: use model.outputs[0]
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.outputs[0]]
    )

    # ✅ Gradient computation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # ❌ Prevent None gradient crash
    if grads is None:
        raise ValueError("❌ Gradients are None (model issue)")

    # ✅ Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    # ✅ Weighted sum
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ✅ ReLU
    heatmap = tf.maximum(heatmap, 0)

    # ✅ Safe normalization
    max_val = tf.reduce_max(heatmap)

    if max_val == 0:
        return np.zeros((128, 128))

    heatmap = heatmap / max_val

    return heatmap.numpy()                                                       