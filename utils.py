import pandas as pd
import ast
import os
import numpy as np
import wfdb
import pickle
import tqdm
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, roc_curve, auc

def load_dataset(path: str, release:bool = False) -> tuple[np.array, pd.DataFrame]:
    path_to_labels = path+'ptbxl_database.csv'
    Y = pd.read_csv(path_to_labels, index_col='ecg_id')

    # scp_codes is the column with statement: likelihood pairs
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    X = load_raw_data_ptbxl(Y, path)

    return X, Y

def load_raw_data_ptbxl(df: pd.DataFrame, path: str) -> np.array:
    if os.path.exists(path + 'raw100.npy'):
        print("Loading raw numpy data")
        data = np.load(path+'raw100.npy', allow_pickle=True)
    else:
        print("Creating and saving raw numpy data")
        data = [wfdb.rdsamp(path+f) for f in tqdm.tqdm(df.filename_lr)]
        data = np.array([signal for signal, meta in data])
        pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    return data

def compute_label_aggregations(df: pd.DataFrame, data_folder: str, task: str)  -> pd.DataFrame:
    # df is ptbxl_database.csv in dataframe form
    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    path_to_scp_statements = data_folder+'scp_statements.csv'
    aggregation_df = pd.read_csv(path_to_scp_statements, index_col=0)
    diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0] # get rows about diagnostic labels

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in diag_agg_df.index:
                c = diag_agg_df.loc[key].diagnostic_class
                if str(c) != 'nan':
                    tmp.append(c)
        return list(set(tmp)) # list of superclasses for a sample's labels


    # Todo: account for different tasks
    if task == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    
    return df


def select_data(data: np.array, labels: pd.DataFrame, task: str):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    # Todo: account for different tasks
    if task == "superdiagnostic":
        # could filter out classes that don't meet a min number of occurrences
        data = data[labels.superdiagnostic_len > 0]
        labels = labels[labels.superdiagnostic_len > 0]
        mlb.fit(labels.superdiagnostic.values)
        y = mlb.transform(labels.superdiagnostic.values)
    
    # save LabelBinarizer
    output_folder = "output/"
    os.makedirs(output_folder, exist_ok=True)
    with open(output_folder+'mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)

    return data, labels, y, mlb


def preprocess_signals(X_train: np.array, X_validation: np.array, X_test: np.array):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:,np.newaxis].astype(float))
    
    # Save Standardizer data
    output_folder = "output/"
    os.makedirs(output_folder, exist_ok=True)
    with open(output_folder+'standard_scaler.pkl', 'wb') as ss_file:
        pickle.dump(ss, ss_file)

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)

def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp

def apply_thresholds(preds, thresholds):
	"""
		apply class-wise thresholds to prediction score in order to get binary format.
		BUT: if no score is above threshold, pick maximum. This is needed due to metric issues.
	"""
	tmp = []
	for p in preds:
		tmp_p = (p > thresholds).astype(int)
		if np.sum(tmp_p) == 0:
			tmp_p[np.argmax(p)] = 1
		tmp.append(tmp_p)
	tmp = np.array(tmp)
	return tmp

def evaluate_experiment(y_true, y_pred, thresholds=None):
    results = {}

    if not thresholds is None:
        # binary predictions
        y_pred_binary = apply_thresholds(y_pred, thresholds)
        # PhysioNet/CinC Challenges metrics
        challenge_scores = challenge_metrics(y_true, y_pred_binary, beta1=2, beta2=2)
        results['F_beta_macro'] = challenge_scores['F_beta_macro']
        results['G_beta_macro'] = challenge_scores['G_beta_macro']

    # label based metric
    results['macro_auc'] = roc_auc_score(y_true, y_pred, average='macro')
    results['class_auc'] = roc_auc_score(y_true, y_pred, average=None)  # Per-class AUC array
    
    # After calculating results['macro_auc'] and results['class_auc']
    df_result = pd.DataFrame({
        'macro_auc': [results['macro_auc']],
        'class_auc': [results['class_auc']]
    })
    return df_result

def challenge_metrics(y_true, y_pred, beta1=2, beta2=2, class_weights=None, single=False):
    f_beta = 0
    g_beta = 0
    if single: # if evaluating single class in case of threshold-optimization
        sample_weights = np.ones(y_true.sum(axis=1).shape)
    else:
        sample_weights = y_true.sum(axis=1)
    for classi in range(y_true.shape[1]):
        y_truei, y_predi = y_true[:,classi], y_pred[:,classi]
        TP, FP, TN, FN = 0.,0.,0.,0.
        for i in range(len(y_predi)):
            sample_weight = sample_weights[i]
            if y_truei[i]==y_predi[i]==1: 
                TP += 1./sample_weight
            if ((y_predi[i]==1) and (y_truei[i]!=y_predi[i])): 
                FP += 1./sample_weight
            if y_truei[i]==y_predi[i]==0: 
                TN += 1./sample_weight
            if ((y_predi[i]==0) and (y_truei[i]!=y_predi[i])): 
                FN += 1./sample_weight 
        f_beta_i = ((1+beta1**2)*TP)/((1+beta1**2)*TP + FP + (beta1**2)*FN)
        g_beta_i = (TP)/(TP+FP+beta2*FN)

        f_beta += f_beta_i
        g_beta += g_beta_i

    return {'F_beta_macro':f_beta/y_true.shape[1], 'G_beta_macro':g_beta/y_true.shape[1]}