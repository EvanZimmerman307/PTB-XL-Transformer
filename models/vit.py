import torch
from torch import nn
from transformers import ViTForImageClassification, ViTConfig
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
import math

class ClassBalancedBCELoss(nn.Module):
    def __init__(self, class_counts, epsilon=1e-7):
        super().__init__()
        """
        class_counts: array/list of length num_classes with #positive examples per class
        """
        self.epsilon = epsilon
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
        total = class_counts.sum()

        # pos_weight = #negative / #positive   (as required by BCEWithLogitsLoss)
        self.pos_weight = (total - class_counts) / (class_counts + epsilon)

        # unwrap to tensor for BCEWithLogits
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
    
    def forward(self, logits, targets):
        return self.bce(logits, targets)

# =========================================================
# Freeze / Unfreeze Helpers
# =========================================================

def freeze_all_but_patch_and_head(model):
    # Freeze everything
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Unfreeze patch embedding
    for param in model.vit.embeddings.patch_embeddings.projection.paramteres():
        param.required_grad = True
    
    # Unfreeze classification head
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    print("Frozen ViT backbone. Training patch embedding + classifier only.")

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    
    print("Unfroze entire ViT model — full finetuning.")

class VIT:

    def __init__(self, model_name: str, new_num_channels: int, num_classes: int, problem_type: str):
        """
        model_name: ViT model name on hugging face
        new_num_channels: number of channels in stacked ECG spectrogram (for 12 leads there are 12 channels)
        num_classes: total number of possible classification classes
        problem_type: "multi_label_classification" for multi-label classification tasks (PTB-XL)
        """
        self.model_name = model_name
        self.num_channels = new_num_channels
        self.num_classes = num_classes
        self.problem_type = problem_type

        # --- 1. Load the pre-trained VIT Model and Modify the Classification Head ---
        # num_labels: Replaces the final classification layer.
        # ignore_mismatched_sizes: Tells the library it's okay that the old classifier (1000 outputs) doesn't match the new one (5 outputs).
        # The new classifier weights are initialized randomly and will be trained from scratch.
        print("Loading model and replacing classification head...")
        self.vit_model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            problem_type=self.problem_type,
            ignore_mismatched_sizes=True
        )

        # --- 2. Perform Patch Embedding Layer Weight Surgery ---
        # Access the original 3-channel convolutional layer (patch embedding)
        original_conv = self.vit_model.vit.embeddings.patch_embeddings.projection
        original_weight = original_conv.weight.data
        original_bias = original_conv.bias.data if original_conv.bias is not None else None

        # Strategy: Weight Averaging and Replication
        # A. Calculate the mean of the weights across the 3 original input channels (dim=1)
        mean_weight = original_weight.mean(dim=1, keepdim=True) # Resulting shape: (768, 1, 16, 16)

        # B. Replicate the mean weight 12 times to match the new number of channels
        # Resulting shape: (768, 12, 16, 16)
        new_weight = mean_weight.repeat(1, self.num_channels, 1, 1)

        # C. Create a new nn.Conv2d layer with num_channels input channels
        new_patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        # D. Load the new 12-channel weights and the original bias into the new layer
        new_patch_embedding.weight.data.copy_(new_weight)
        if original_bias is not None:
            new_patch_embedding.bias.data.copy_(original_bias)
        
        # E. Replace the old layer with the new one in the model architecture
        self.vit_model.vit.embeddings.patch_embeddings.projection = new_patch_embedding

        # F. Update the model's configuration
        self.vit_model.config.num_channels = self.num_channels

        # --- 3. Verification ---
        print("\n--- Verification ---")
        print(f"New number of model output classes (Classification Head): {self.vit_model.config.num_labels}")
        print(f"New patch embedding layer input channels: {self.vit_model.vit.embeddings.patch_embeddings.projection.in_channels}")
        print(f"New patch embedding weight shape: {self.vit_model.vit.embeddings.patch_embeddings.projection.weight.shape}")
    
    

    

