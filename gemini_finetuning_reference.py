import torch
from torch import nn
from transformers import ViTForImageClassification, ViTConfig
from collections import OrderedDict

# --- Configuration Variables ---
MODEL_NAME = "google/vit-base-patch16-224"
NEW_NUM_CHANNELS = 12   # For your 12-lead ECG spectrograms
YOUR_NUM_CLASSES = 5    # Replace with the actual number of classes in your PTB-XL task

# --- 1. Load the Model and Modify the Classification Head ---
# Load the pre-trained ViT model.
# - num_labels: Replaces the final classification layer.
# - ignore_mismatched_sizes: Tells the library it's okay that the old
#                            classifier (1000 outputs) doesn't match the new one (5 outputs).
# The new classifier weights are initialized randomly and will be trained from scratch.
print("Loading model and replacing classification head...")
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=YOUR_NUM_CLASSES,
    ignore_mismatched_sizes=True
)

# --- 2. Perform Patch Embedding Layer Weight Surgery ---

# Access the original 3-channel convolutional layer (patch embedding)
original_conv = model.vit.embeddings.patch_embeddings.projection
original_weight = original_conv.weight.data
original_bias = original_conv.bias.data if original_conv.bias is not None else None

# Check the original shape: typically (768, 3, 16, 16) -> (out, in, kernel_h, kernel_w)
print(f"Original patch embedding weight shape: {original_weight.shape}")

# Strategy: Weight Averaging and Replication
# A. Calculate the mean of the weights across the 3 original input channels (dim=1)
# Resulting shape: (768, 1, 16, 16)
mean_weight = original_weight.mean(dim=1, keepdim=True)

# B. Replicate the mean weight 12 times to match the new number of channels
# Resulting shape: (768, 12, 16, 16)
new_weight = mean_weight.repeat(1, NEW_NUM_CHANNELS, 1, 1)

# C. Create a new nn.Conv2d layer with 12 input channels
new_conv = nn.Conv2d(
    in_channels=NEW_NUM_CHANNELS,
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias is not None # Keep bias status consistent
)

# D. Load the new 12-channel weights and the original bias into the new layer
new_conv.weight.data.copy_(new_weight)
if original_bias is not None:
    new_conv.bias.data.copy_(original_bias)

# E. Replace the old layer with the new one in the model architecture
model.vit.embeddings.patch_embeddings.projection = new_conv

# F. Update the model's configuration
model.config.num_channels = NEW_NUM_CHANNELS

# --- 3. Verification ---
print("\n--- Verification ---")
print(f"New number of model output classes (Classification Head): {model.config.num_labels}")
print(f"New patch embedding layer input channels: {model.vit.embeddings.patch_embeddings.projection.in_channels}")
print(f"New patch embedding weight shape: {model.vit.embeddings.patch_embeddings.projection.weight.shape}")

# The model is now ready for fine-tuning on your 12-channel, 5-class task.