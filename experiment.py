import os
import pickle
import pandas as pd
import numpy as np
import utils
from models.ecg_hybrid_net import ECGHybridNet
from models.resnet_transformer import ResNetTransformer

class Experiment():

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

    def prepare(self):
        self.data, self.raw_labels = utils.load_dataset(self.data_folder)

        self.labels = utils.compute_label_aggregations(self.raw_labels, self.data_folder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task)
        self.input_shape = self.data[0].shape
        
        # 10th fold for testing
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # Preprocess signal data
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test)
        self.n_classes = self.y_train.shape[1]

        # save train and test labels
        output_folder = "output/"
        os.makedirs(output_folder + self.experiment_name + '/data/', exist_ok=True)
        self.y_train.dump(output_folder + self.experiment_name + '/data/y_train.npy')
        self.y_val.dump(output_folder + self.experiment_name + '/data/y_val.npy')
        self.y_test.dump(output_folder + self.experiment_name + '/data/y_test.npy')

    def train(self):
        output_folder = "output/"
        model_path = output_folder+self.experiment_name+'/model/'

        # create folder for model outputs
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(model_path+'results/'):
            os.makedirs(model_path+'results/')
        
        n_classes = self.Y.shape[1]

        # self.model = ECGHybridNet(input_dim=self.input_shape[1], num_classes=n_classes)
        self.model = ResNetTransformer(input_dim=self.input_shape[1], num_classes=n_classes)
        self.model.fit(self.X_train, self.y_train, self.X_val, self.y_val, model_path)
        
    def predict(self):
        # predict and dump
        output_folder = "output/"
        model_path = output_folder+self.experiment_name+'/model/'
        self.model.predict(self.X_train).dump(model_path+'y_train_pred.npy')
        self.model.predict(self.X_val).dump(model_path+'y_val_pred.npy')
        self.model.predict(self.X_test).dump(model_path+'y_test_pred.npy')

    def evaluate(self):
        output_folder = "output/"
        model_path = output_folder+self.experiment_name+'/model/'
        y_train_pred = np.load(model_path+'y_train_pred.npy', allow_pickle=True)
        y_test_pred = np.load(model_path+'y_test_pred.npy', allow_pickle=True)
        print(utils.evaluate_experiment(self.y_train, y_train_pred))
        print(utils.evaluate_experiment(self.y_test, y_test_pred))
