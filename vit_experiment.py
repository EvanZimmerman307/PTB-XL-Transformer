import os
import pickle
import pandas as pd
import numpy as np
import utils
from models.vit import VIT
from vit_ecg_dataset import ECGDataset
from torch.utils.data import Dataset, DataLoader

class VITExperiment():

    def __init__(self, experiment_name: str,  task: str, data_folder: str, sampling_frequency: int = 100):
        self.data = None
        self.raw_labels = None
        self.data_folder = data_folder
        self.fs = sampling_frequency
        self.task = task
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.X_train = None
        self.y_train = None
        self.test_fold = 10
        self.val_fold = 9
        self.train_fold = 8
        self.experiment_name = experiment_name
        self.input_shape = None
        self.model = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
    
    def prepare(self):
        print("Preparing raw ECG data by loading to numpy format and splitting into train/val/test splits")
        self.data, self.raw_labels = utils.load_dataset(self.data_folder)

        self.labels = utils.compute_label_aggregations(self.raw_labels, self.data_folder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task)
        self.input_shape = self.data[0].shape
        
        # 10th fold for testing
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.X_test = np.swapaxes(self.X_test, self.X_test.ndim - 2, self.X_test.ndim - 1)
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.X_val = np.swapaxes(self.X_val, self.X_val.ndim - 2, self.X_val.ndim - 1)
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.X_train = np.swapaxes(self.X_train, self.X_train.ndim - 2, self.X_train.ndim - 1)
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]
        print(f"X_train shape: {self.X_train.shape}")
    
    def create_dataloaders(self):
        print("Creating PyTorch Dataloaders for train/val/test splits")

        # Generate PyTorch datasets for the train, val, and test splits from the raw data in numpy format
        train_dataset = ECGDataset(self.X_train, self.y_train)
        print(f"Train Dataset Size: {len(train_dataset)} samples")
        val_dataset = ECGDataset(self.X_val, self.y_val)
        print(f"Val Dataset Size: {len(val_dataset)} samples")
        test_dataset = ECGDataset(self.X_test, self.y_test)
        print(f"Test Dataset Size: {len(test_dataset)} samples")

        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=4
        )
        print(f"Train DataLoader Batches: {len(self.train_dataloader)} batches")

        self.val_dataloader = DataLoader(
            val_dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=4
        )
        print(f"Val DataLoader Batches: {len(self.val_dataloader)} batches")

        self.test_dataloader = DataLoader(
            test_dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=4
        )
        print(f"Test DataLoader Batches: {len(self.test_dataloader)} batches")
    
    def fetch_one_batch(self):
        print("\n--- Testing DataLoader (On-the-fly Preprocessing) ---")
    
        for batch_idx, (spectrograms, labels) in enumerate(self.train_dataloader):
            if batch_idx == 0:
                print(f"Batch {batch_idx+1}:")
                print("Expected shape: (Batch_Size, C, H, W) -> (32, 12, 224, 224)")
                print(f"Spectrograms Batch Shape: {spectrograms.shape}")
                print("Expected shape: (Batch_Size, Num_Classes) -> (32, 5)")
                print(f"Labels Batch Shape: {labels.shape}")
                print(f"Data Type: {spectrograms.dtype}")
                print("Successfully processed the first batch on-the-fly.")
                break
    
    def train(self):
        output_folder = "output/"
        model_path = output_folder+self.experiment_name+'/model/'

        # create folder for model outputs
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(model_path+'results/'):
            os.makedirs(model_path+'results/')
        
        n_classes = self.Y.shape[1]
        MODEL_NAME = "google/vit-base-patch16-224"
        NEW_NUM_CHANNELS = 12   # For your 12-lead ECG spectrograms
        NUM_CLASSES = n_classes    # Replace with the actual number of classes in your PTB-XL task
        PROBLEM_TYPE = "multi_label_classification"
        vit_params = {
        "model_name": MODEL_NAME,
        "num_channels": NEW_NUM_CHANNELS,
        "num_classes": NUM_CLASSES,
        "problem_type": PROBLEM_TYPE
        }
        print(f"Attempting to initialize ViT with params: {vit_params}")
        self.model = VIT(
            model_name=MODEL_NAME,
            new_num_channels=NEW_NUM_CHANNELS,
            num_classes=NUM_CLASSES,
            problem_type=PROBLEM_TYPE
        )
        print(f"Initialized Vit with params: model_name: {self.model.model_name}, num_channels: {self.model.num_channels}, num_classes: {self.model.num_classes}, problem_type: {self.model.problem_type}")

    


