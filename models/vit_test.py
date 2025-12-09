from vit import VIT

def main():

    MODEL_NAME = "google/vit-base-patch16-224"
    NEW_NUM_CHANNELS = 12   # For your 12-lead ECG spectrograms
    NUM_CLASSES = 5    # Replace with the actual number of classes in your PTB-XL task
    PROBLEM_TYPE = "multi_label_classification"
    vit_params = {
        "model_name": MODEL_NAME,
        "num_channels": NEW_NUM_CHANNELS,
        "num_classes": NUM_CLASSES,
        "problem_type": PROBLEM_TYPE
    }
    print(f"Attempting to initialize ViT with params: {vit_params}")
    vit = VIT(
        model_name=MODEL_NAME,
        new_num_channels=NEW_NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        problem_type=PROBLEM_TYPE
    )
    print(f"Initialized Vit with params: model_name: {vit.model_name}, num_channels: {vit.num_channels}, num_classes: {vit.num_classes}, problem_type: {vit.problem_type}")

if __name__ == "__main__":
    main()