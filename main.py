import random
import torch
import numpy as np
from experiment import Experiment

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_folder = "ptb-xl/"
"""
Experiment 1: ecg_hybrid_net, train macro_auc: 0.949275, test macro_auc: 0.907747
Experiment 2: hybrid_net_class_balanced_loss train macro_auc: 0.958235, test macro_auc: 0.910611
Experiment 3: hybrid net + class balanced loss + positional encodings
Train evaluation:
   macro_auc                                          class_auc
0   0.943679  [0.9320465235795772, 0.9431944002735974, 0.941...
Test evaluation:
   macro_auc                                          class_auc
0   0.906574  [0.8926393094212182, 0.8879320063130093, 0.894...
Experiment 4: ditch positional encodings, increase dim and layers and dropout, decrease lr and decay
 macro_auc                                          class_auc
0   0.954212  [0.9448535162572305, 0.9558127844946374, 0.950...
   macro_auc                                          class_auc
0   0.910634  [0.8945389833469197, 0.8990280864495765, 0.895...
Experiment 5: Extra conv layer at 128 channels & Max Pool = deeper features = better features for the transformer
macro_auc                                          class_auc
0   0.964925  [0.9657599443570567, 0.9654953582899977, 0.963...
   macro_auc                                          class_auc
0   0.916371  [0.9085148092077171, 0.8942087802364157, 0.908...
Experiment 6: Reduce model params to improve training stability
Best model saved to output/ecg_hybrid_net_exp6_maxpool_small/model/best.pth
   macro_auc                                          class_auc
0   0.970027  [0.9703815112925958, 0.9686367766727051, 0.970...
   macro_auc                                          class_auc
0   0.913247  [0.9034271767400333, 0.8890271201726414, 0.909...
Experiment 7: Add back positional encoding -> didn't do anything tbh
macro_auc                                          class_auc
0   0.965072  [0.9658338334867284, 0.9679896319349522, 0.963...
   macro_auc                                          class_auc
0   0.913576  [0.8996526969061759, 0.8964835571874898, 0.905...
Experiment 8: Add attention pooling (learns to weight which time steps matter most for each prediction, rather than averaging all positions equally.) -> worse
   macro_auc                                          class_auc
0   0.947941  [0.9494131884210295, 0.941687565149262, 0.9452...
   macro_auc                                          class_auc
0   0.910372  [0.9127496506346803, 0.8878313524656167, 0.894...
Experiment 9: Remove positional encoding and keep attentionpool to isolate
   macro_auc                                          class_auc
0   0.950735  [0.9525351692541814, 0.9469618822714696, 0.946...
   macro_auc                                          class_auc
0   0.912172  [0.9109021388921237, 0.8931398363771056, 0.892...
Experiment 10: Resnet
 macro_auc                                          class_auc
0   0.981557  [0.9833567940384603, 0.9865742734305167, 0.978...
   macro_auc                                          class_auc
0   0.901265  [0.8803720983269283, 0.8772023061809515, 0.893...
Experiment 11: Resnet small + dropout
 macro_auc                                          class_auc
0   0.954085  [0.9551087505610396, 0.9492076446425962, 0.952...
   macro_auc                                          class_auc
0    0.91443  [0.9067279511664921, 0.8932445163783939, 0.902...
Experiment 12: resnet_medium (adding bigger hidden dim and transformer layers)
   macro_auc                                          class_auc
0    0.96492  [0.9666443024498073, 0.9627997023730217, 0.965...
   macro_auc                                          class_auc
0   0.917024  [0.9147876149994179, 0.8928640448352497, 0.906...
Experiment 13: resnet_medium (adding bigger hidden dim and transformer layers) + gaussian noise + random scaling
   macro_auc                                          class_auc
0   0.954292  [0.9551121012116361, 0.9491638741143208, 0.952...
   macro_auc                                          class_auc
0    0.91286  [0.9037789682077559, 0.888469497858086, 0.9032...
Experiment 14 (Last Augmentation): resnet_medium (adding bigger hidden dim and transformer layers) + gaussian noise + random scaling + wander
Best model saved to output/exp14_hybrid_resnet_medium_noise_scale_wander/model/best.pth
   macro_auc                                          class_auc
0   0.957393  [0.9584630043207465, 0.9540695159872012, 0.955...
   macro_auc                                          class_auc
0   0.913614  [0.9052310178176314, 0.8868107224530551, 0.906...
"""
experiment = Experiment('exp14_hybrid_resnet_medium_noise_scale_wander', 'superdiagnostic', data_folder)
experiment.prepare()
experiment.train()
experiment.predict()
experiment.evaluate()
