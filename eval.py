import torch
import numpy as np
from models.ecg_hybrid_net import ECGHybridNet
from experiment import Experiment
import utils
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Paths
experiment_name = 'ecg_hybrid_net_exp4_large'
data_folder = "ptb-xl/"
output_folder = "output/"
model_path = output_folder + experiment_name + '/model/'

# Create experiment but only prepare to load data
experiment = Experiment(experiment_name, 'superdiagnostic', data_folder)
experiment.prepare()

# Evaluate
output_folder = "output/"
model_path = output_folder+experiment_name+'/model/'
print("Train evaluation:")
predictions_train = np.load(model_path+'y_train_pred.npy', allow_pickle=True)
predictions_test = np.load(model_path+'y_test_pred.npy', allow_pickle=True)
train_res = utils.evaluate_experiment(experiment.y_train, predictions_train)
print(train_res)
train_res.to_csv(output_folder+experiment_name+'/model/'+'results/train_res.csv')

print("\nTest evaluation:")
test_res = utils.evaluate_experiment(experiment.y_test, predictions_test)
print(test_res)
test_res.to_csv(output_folder+experiment_name+'/model/'+'results/test_res.csv')