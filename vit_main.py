import random
import torch
import numpy as np
from vit_experiment import VITExperiment

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_FOLDER = "ptb-xl/"

def vit_main():
    experiment = VITExperiment('', 'superdiagnostic', DATA_FOLDER)
    experiment.prepare()
    experiment.create_dataloaders()
    # experiment.fetch_one_batch() uncomment to test the data loader is working
    experiment.train()


if __name__ == "__main__":
    vit_main()