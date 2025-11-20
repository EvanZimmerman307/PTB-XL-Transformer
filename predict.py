import torch
import numpy as np
from models.ecg_hybrid_net import ECGHybridNet
from experiment import Experiment
import utils
import random

# Set seeds again for consistency, though not needed for inference
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Paths
experiment_name = 'ecg_hybrid_net'
data_folder = "ptb-xl/"
output_folder = "output/"
model_path = output_folder + experiment_name + '/model/'

# Create experiment but only prepare to load data
experiment = Experiment(experiment_name, 'superdiagnostic', data_folder)
experiment.prepare()

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_classes = experiment.y_train.shape[1]
input_dim = experiment.input_shape[1]
model = ECGHybridNet(input_dim=input_dim, num_classes=n_classes)
model.load_state_dict(torch.load(model_path + 'best.pth', weights_only=True))
model.to(device)
model.eval()

# Predict and save
predictions_train = model.predict(experiment.X_train)
np.save(model_path + 'y_train_pred.npy', predictions_train)

predictions_val = model.predict(experiment.X_val)
np.save(model_path + 'y_val_pred.npy', predictions_val)

predictions_test = model.predict(experiment.X_test)
np.save(model_path + 'y_test_pred.npy', predictions_test)

print("Predictions saved.")

# Evaluate
print("Train evaluation:")
print(utils.evaluate_experiment(experiment.y_train, predictions_train))
print("\nTest evaluation:")
print(utils.evaluate_experiment(experiment.y_test, predictions_test))
