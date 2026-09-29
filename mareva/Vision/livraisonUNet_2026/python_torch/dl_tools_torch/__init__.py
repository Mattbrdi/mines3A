"""PyTorch port of the dl_tools package.

Framework-independent modules (random_image_generator, eval) are kept close
to the original; the Keras-specific modules (modular_unet, keras_custom_loss,
keras_image2image, tools) have been rewritten for PyTorch and are suffixed
with `_torch`.
"""
