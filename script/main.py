import math
import os
import sys
import time
import configparser

# import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Agg')
from argparse import ArgumentParser
from functools import reduce
from keras import optimizers
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from utils.callbacks import AccHistoryPlot, EarlyStopping
from utils.basic_modules import *
from utils.feature_encoder import get_pro_motif_fea, get_rna_motif_fea, get_pro_pc_fea, get_rna_pc_fea
from utils.sequence_encoder import ProEncoder, RNAEncoder
from utils.stacked_auto_encoder import train_auto_encoder


# default program settings
DATA_SET = 'NPInter'
TIME_FORMAT = "-%y-%m-%d-%H-%M-%S"


WINDOW_P_UPLIMIT = 3  # 联体
WINDOW_P_STRUCT_UPLIMIT = 3  # 结构
WINDOW_R_UPLIMIT = 4
WINDOW_R_STRUCT_UPLIMIT = 4
VECTOR_REPETITION_CNN = 1
RANDOM_SEED = 1
K_FOLD = 5
BATCH_SIZE = 150
FIRST_TRAIN_EPOCHS = [25]
SECOND_TRAIN_EPOCHS = [10]
PATIENCES = [10]
FIRST_OPTIMIZER = 'adam'
SECOND_OPTIMIZER = 'sgd'
SGD_LEARNING_RATE = 0.005
ADAM_LEARNING_RATE = 0.001
FREEZE_SUB_MODELS = True
CODING_FREQUENCY = True
MONITOR = 'acc'
MIN_DELTA = 0.0
SHUFFLE = True
VERBOSE = 2


# get the path of RPI-EDLCN.py
script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
parent_dir = os.path.dirname(script_dir)
# set paths of data, results and program parameters
DATA_BASE_PATH = parent_dir + '/data/'
RESULT_BASE_PATH = parent_dir + '/result/'
INI_PATH = script_dir + '/utils/data_set_settings.ini'

metrics_whole = {'RPI-EDLCN': np.zeros(7)}
parser = ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help='The dataset you want to process.')
args = parser.parse_args()
if args.dataset != None:
    DATA_SET = args.dataset
print("Dataset: %s" % DATA_SET)

# set result save path
result_save_path = RESULT_BASE_PATH + DATA_SET + "/" + DATA_SET + time.strftime(TIME_FORMAT, time.localtime()) + "/"
if not os.path.exists(result_save_path):
    os.mkdir(result_save_path)
out = open(result_save_path + 'result.txt', 'w')


def read_data_pair(path):
    pos_pairs = []
    neg_pairs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            p, r, label = line.split('\t')
            if label == '1':
                pos_pairs.append((p, r))
            elif label == '0':
                neg_pairs.append((p, r))
    return pos_pairs, neg_pairs


def read_data_seq(path):
    seq_dict = {}
    with open(path, 'r') as f:
        name = ''
        for line in f:
            line = line.strip()
            if line[0] == '>':
                name = line[1:]
                seq_dict[name] = ''
            else:
                if line.startswith('XXX'):
                    seq_dict.pop(name)
                else:
                    seq_dict[name] = line
    return seq_dict


# calculate the seven metrics of Acc, Sn, Sp, Precision, F1_measure, MCC and AUC
def calc_metrics(y_label, y_proba):
    con_matrix = confusion_matrix(y_label, [1 if x >= 0.5 else 0 for x in y_proba])
    print(con_matrix)
    TN = float(con_matrix[0][0])
    FP = float(con_matrix[0][1])
    FN = float(con_matrix[1][0])
    TP = float(con_matrix[1][1])
    P = TP + FN
    N = TN + FP
    Sn = TP / P if P > 0 else 0
    Sp = TN / N if N > 0 else 0
    Acc = (TP + TN) / (P + N) if (P + N) > 0 else 0
    Pre = (TP) / (TP + FP) if (TP+FP) > 0 else 0
    F1_measure = (2*Sn*Pre)/(Sn+Pre)
    MCC = 0
    tmp = math.sqrt((TP + FP) * (TP + FN)) * math.sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp
    fpr, tpr, thresholds = roc_curve(y_label, y_proba)
    AUC = auc(fpr, tpr)
    return Acc, Sn, Sp, Pre, F1_measure, MCC, AUC


def load_data(data_set):
    pro_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_protein_seq.fa')
    rna_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_rna_seq.fa')
    pro_structs = read_data_seq(DATA_BASE_PATH + "structure/" + data_set + '_protein_struct.fa')
    rna_structs = read_data_seq(DATA_BASE_PATH + "structure/" + data_set + '_rna_struct.fa')
    pos_pairs, neg_pairs = read_data_pair(DATA_BASE_PATH + data_set + '_pairs.txt')

    return pos_pairs, neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs


def coding_pairs(pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, PE, RE, kind):
    samples = []
    for pr in pairs:
        if pr[0] in pro_seqs and pr[1] in rna_seqs and pr[0] in pro_structs and pr[1] in rna_structs:
            p_seq = pro_seqs[pr[0]]  # protein sequence
            r_seq = rna_seqs[pr[1]]  # rna sequence
            p_struct = pro_structs[pr[0]]  # protein structure
            r_struct = rna_structs[pr[1]]  # rna structure

            p_motif_fea = get_pro_motif_fea(p_seq)  # protein motif feature
            r_motif_fea = get_rna_motif_fea(r_seq)  # rna motif feature
            p_pc_fea = get_pro_pc_fea(p_seq, fourier_len=10)  # protein physicochemical properties feature
            r_pc_fea = get_rna_pc_fea(r_seq, fourier_len=10)  # rna physicochemical properties feature
            p_fea = np.append(p_motif_fea, p_pc_fea)
            r_fea = np.append(r_motif_fea, r_pc_fea)

            p_conjoint = PE.encode_conjoint(p_seq)  # protein sequence feature
            r_conjoint = RE.encode_conjoint(r_seq)  # rna sequence feature
            p_conjoint_struct = PE.encode_conjoint_struct(p_seq, p_struct)   # protein sequence feature + struct feature
            r_conjoint_struct = RE.encode_conjoint_struct(r_seq, r_struct)   # rna sequence feature + struct feature
            p_conjoint_struct_fea1 = np.append(p_conjoint_struct, p_fea)  # protein sequence feature + struct feature + motif feature + physicochemical properties feature
            r_conjoint_struct_fea1 = np.append(r_conjoint_struct, r_fea)  # rna sequence feature  + struct feature  + motif feature + physicochemical properties feature
            p_conjoint_struct_fea2 = p_conjoint_struct_fea1
            r_conjoint_struct_fea2 = r_conjoint_struct_fea1
            p_conjoint_struct_fea3 = p_conjoint_struct_fea1
            r_conjoint_struct_fea3 = r_conjoint_struct_fea1

            if p_conjoint is 'Error':
                print('Skip {} in pair {} according to conjoint coding process.'.format(pr[0], pr))
            elif r_conjoint is 'Error':
                print('Skip {} in pair {} according to conjoint coding process.'.format(pr[1], pr))
            elif p_conjoint_struct is 'Error':
                print('Skip {} in pair {} according to conjoint_struct coding process.'.format(pr[0], pr))
            elif r_conjoint_struct is 'Error':
                print('Skip {} in pair {} according to conjoint_struct coding process.'.format(pr[1], pr))

            else:
                samples.append([[p_conjoint, r_conjoint],
                                [p_conjoint_struct, r_conjoint_struct],
                                [p_conjoint_struct_fea1, r_conjoint_struct_fea1],
                                [p_conjoint_struct_fea2, r_conjoint_struct_fea2],
                                [p_conjoint_struct_fea3, r_conjoint_struct_fea3],
                                kind])
        else:
            print('Skip pair {} according to sequence dictionary.'.format(pr))
    return samples


def standardization(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def pre_process_data(samples, samples_pred=None):
    # np.random.shuffle(samples)

    p_conjoint = np.array([x[0][0] for x in samples])
    r_conjoint = np.array([x[0][1] for x in samples])
    p_conjoint_struct = np.array([x[1][0] for x in samples])
    r_conjoint_struct = np.array([x[1][1] for x in samples])
    p_conjoint_struct_fea1 = np.array([x[2][0] for x in samples])
    r_conjoint_struct_fea1 = np.array([x[2][1] for x in samples])
    p_conjoint_struct_fea2 = np.array([x[3][0] for x in samples])
    r_conjoint_struct_fea2 = np.array([x[3][1] for x in samples])
    p_conjoint_struct_fea3 = np.array([x[4][0] for x in samples])
    r_conjoint_struct_fea3 = np.array([x[4][1] for x in samples])
    y_samples = np.array([x[5] for x in samples])

    p_conjoint, scaler_p = standardization(p_conjoint)
    r_conjoint, scaler_r = standardization(r_conjoint)
    p_conjoint_struct, scaler_p_struct = standardization(p_conjoint_struct)
    r_conjoint_struct, scaler_r_struct = standardization(r_conjoint_struct)
    p_conjoint_struct_fea1, scaler_p_struct_fea1 = standardization(p_conjoint_struct_fea1)
    r_conjoint_struct_fea1, scaler_r_struct_fea1 = standardization(r_conjoint_struct_fea1)
    p_conjoint_struct_fea2, scaler_p_struct_fea2 = standardization(p_conjoint_struct_fea2)
    r_conjoint_struct_fea2, scaler_r_struct_fea2 = standardization(r_conjoint_struct_fea2)
    p_conjoint_struct_fea3, scaler_p_struct_fea3 = standardization(p_conjoint_struct_fea3)
    r_conjoint_struct_fea3, scaler_r_struct_fea3 = standardization(r_conjoint_struct_fea3)

    p_conjoint_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint])
    r_conjoint_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint])
    p_conjoint_struct_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct])
    r_conjoint_struct_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct])
    p_conjoint_struct_fea1_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct_fea1])
    r_conjoint_struct_fea1_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct_fea1])
    p_conjoint_struct_fea2_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct_fea2])
    r_conjoint_struct_fea2_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct_fea2])
    p_conjoint_struct_fea3_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct_fea3])
    r_conjoint_struct_fea3_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct_fea3])

    p_ctf_len = 7 ** WINDOW_P_UPLIMIT
    r_ctf_len = 4 ** WINDOW_R_UPLIMIT
    p_conjoint_previous = np.array([x[-p_ctf_len:] for x in p_conjoint])
    r_conjoint_previous = np.array([x[-r_ctf_len:] for x in r_conjoint])

    X_samples = [[p_conjoint, r_conjoint],
                 [p_conjoint_struct, r_conjoint_struct],
                 [p_conjoint_struct_fea1, r_conjoint_struct_fea1],
                 [p_conjoint_struct_fea2, r_conjoint_struct_fea2],
                 [p_conjoint_struct_fea3, r_conjoint_struct_fea3],
                 [p_conjoint_cnn, r_conjoint_cnn],
                 [p_conjoint_struct_cnn, r_conjoint_struct_cnn],
                 [p_conjoint_struct_fea1_cnn, r_conjoint_struct_fea1_cnn],
                 [p_conjoint_struct_fea2_cnn, r_conjoint_struct_fea2_cnn],
                 [p_conjoint_struct_fea3_cnn, r_conjoint_struct_fea3_cnn],
                 [p_conjoint_previous, r_conjoint_previous]
                 ]

    if samples_pred:
        # np.random.shuffle(samples_pred)

        p_conjoint_pred = np.array([x[0][0] for x in samples_pred])
        r_conjoint_pred = np.array([x[0][1] for x in samples_pred])
        p_conjoint_struct_pred = np.array([x[1][0] for x in samples_pred])
        r_conjoint_struct_pred = np.array([x[1][1] for x in samples_pred])
        p_conjoint_struct_fea1_pred = np.array([x[2][0] for x in samples_pred])
        r_conjoint_struct_fea1_pred = np.array([x[2][1] for x in samples_pred])
        p_conjoint_struct_fea2_pred = np.array([x[3][0] for x in samples_pred])
        r_conjoint_struct_fea2_pred = np.array([x[3][1] for x in samples_pred])
        p_conjoint_struct_fea3_pred = np.array([x[4][0] for x in samples_pred])
        r_conjoint_struct_fea3_pred = np.array([x[4][1] for x in samples_pred])
        y_samples_pred = np.array([x[5] for x in samples_pred])

        p_conjoint_pred = scaler_p.transform(p_conjoint_pred)
        r_conjoint_pred = scaler_r.transform(r_conjoint_pred)
        p_conjoint_struct_pred = scaler_p_struct.transform(p_conjoint_struct_pred)
        r_conjoint_struct_pred = scaler_r_struct.transform(r_conjoint_struct_pred)
        p_conjoint_struct_fea1_pred = scaler_p_struct_fea1.transform(p_conjoint_struct_fea1_pred)
        r_conjoint_struct_fea1_pred = scaler_r_struct_fea1.transform(r_conjoint_struct_fea1_pred)
        p_conjoint_struct_fea2_pred = scaler_p_struct_fea2.transform(p_conjoint_struct_fea2_pred)
        r_conjoint_struct_fea2_pred = scaler_r_struct_fea2.transform(r_conjoint_struct_fea2_pred)
        p_conjoint_struct_fea3_pred = scaler_p_struct_fea3.transform(p_conjoint_struct_fea3_pred)
        r_conjoint_struct_fea3_pred = scaler_r_struct_fea3.transform(r_conjoint_struct_fea3_pred)

        p_conjoint_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_pred])
        r_conjoint_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_pred])
        p_conjoint_struct_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct_pred])
        r_conjoint_struct_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct_pred])
        p_conjoint_struct_fea1_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct_fea1_pred])
        r_conjoint_struct_fea1_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct_fea1_pred])
        p_conjoint_struct_fea2_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct_fea2_pred])
        r_conjoint_struct_fea2_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct_fea2_pred])
        p_conjoint_struct_fea3_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct_fea3_pred])
        r_conjoint_struct_fea3_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct_fea3_pred])
        p_conjoint_previous_pred = np.array([x[-p_ctf_len:] for x in p_conjoint_pred])
        r_conjoint_previous_pred = np.array([x[-r_ctf_len:] for x in r_conjoint_pred])

        X_samples_pred = [[p_conjoint_pred, r_conjoint_pred],
                          [p_conjoint_struct_pred, r_conjoint_struct_pred],
                          [p_conjoint_struct_fea1_pred, r_conjoint_struct_fea1_pred],
                          [p_conjoint_struct_fea2_pred, r_conjoint_struct_fea2_pred],
                          [p_conjoint_struct_fea3_pred, r_conjoint_struct_fea3_pred],
                          [p_conjoint_cnn_pred, r_conjoint_cnn_pred],
                          [p_conjoint_struct_cnn_pred, r_conjoint_struct_cnn_pred],
                          [p_conjoint_struct_fea1_cnn_pred, r_conjoint_struct_fea1_cnn_pred],
                          [p_conjoint_struct_fea2_cnn_pred, r_conjoint_struct_fea2_cnn_pred],
                          [p_conjoint_struct_fea3_cnn_pred, r_conjoint_struct_fea3_cnn_pred],
                          [p_conjoint_previous_pred, r_conjoint_previous_pred]
                          ]

        return X_samples, y_samples, X_samples_pred, y_samples_pred

    else:
        return X_samples, y_samples


def sum_power(num, bottom, top):
    return reduce(lambda x, y: x + y, map(lambda x: num ** x, range(bottom, top + 1)))


def get_callback_list(patience, result_path, stage, fold, X_test, y_test):
    earlystopping = EarlyStopping(monitor=MONITOR, min_delta=MIN_DELTA, patience=patience, verbose=1,
                                  mode='auto', restore_best_weights=True)
    acchistory = AccHistoryPlot([stage, fold], [X_test, y_test], data_name=DATA_SET,
                                result_save_path=result_path, validate=0, plot_epoch_gap=10)

    return [acchistory, earlystopping]


def get_optimizer(opt_name):
    if opt_name == 'sgd':
        return optimizers.sgd(lr=SGD_LEARNING_RATE, momentum=0.5)
    elif opt_name == 'adam':
        return optimizers.adam(lr=ADAM_LEARNING_RATE)
    else:
        return opt_name


def control_model_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable

def get_auto_encoders(X_train, X_test, batch_size=BATCH_SIZE):
    encoders_protein, decoders_protein, train_tmp_p, test_tmp_p = train_auto_encoder(
        X_train=X_train[0],
        X_test=X_test[0],
        layers=[X_train[0].shape[1], 256, 128, 64], batch_size=batch_size)
    encoders_rna, decoders_rna, train_tmp_r, test_tmp_r = train_auto_encoder(
        X_train=X_train[1],
        X_test=X_test[1],
        layers=[X_train[1].shape[1], 256, 128, 64], batch_size=batch_size)
    return encoders_protein, encoders_rna

# load data settings
if DATA_SET in ['RPI1807', 'RPI2241', 'NPInter']:
    config = configparser.ConfigParser()
    config.read(INI_PATH)
    WINDOW_P_UPLIMIT = config.getint(DATA_SET, 'WINDOW_P_UPLIMIT')
    WINDOW_P_STRUCT_UPLIMIT = config.getint(DATA_SET, 'WINDOW_P_STRUCT_UPLIMIT')
    WINDOW_R_UPLIMIT = config.getint(DATA_SET, 'WINDOW_R_UPLIMIT')
    WINDOW_R_STRUCT_UPLIMIT = config.getint(DATA_SET, 'WINDOW_R_STRUCT_UPLIMIT')
    VECTOR_REPETITION_CNN = config.getint(DATA_SET, 'VECTOR_REPETITION_CNN')
    RANDOM_SEED = config.getint(DATA_SET, 'RANDOM_SEED')
    K_FOLD = config.getint(DATA_SET, 'K_FOLD')
    BATCH_SIZE = config.getint(DATA_SET, 'BATCH_SIZE')
    PATIENCES = [int(x) for x in config.get(DATA_SET, 'PATIENCES').replace('[', '').replace(']', '').split(',')]
    FIRST_TRAIN_EPOCHS = [int(x) for x in
                          config.get(DATA_SET, 'FIRST_TRAIN_EPOCHS').replace('[', '').replace(']', '').split(',')]
    SECOND_TRAIN_EPOCHS = [int(x) for x in
                           config.get(DATA_SET, 'SECOND_TRAIN_EPOCHS').replace('[', '').replace(']', '').split(',')]
    FIRST_OPTIMIZER = config.get(DATA_SET, 'FIRST_OPTIMIZER')
    SECOND_OPTIMIZER = config.get(DATA_SET, 'SECOND_OPTIMIZER')
    SGD_LEARNING_RATE = config.getfloat(DATA_SET, 'SGD_LEARNING_RATE')
    ADAM_LEARNING_RATE = config.getfloat(DATA_SET, 'ADAM_LEARNING_RATE')
    FREEZE_SUB_MODELS = config.getboolean(DATA_SET, 'FREEZE_SUB_MODELS')
    CODING_FREQUENCY = config.getboolean(DATA_SET, 'CODING_FREQUENCY')
    MONITOR = config.get(DATA_SET, 'MONITOR')
    MIN_DELTA = config.getfloat(DATA_SET, 'MIN_DELTA')

# write program parameter settings to result file
settings = (
    """# Analyze data set {}\n
Program parameters:
WINDOW_P_UPLIMIT = {},
WINDOW_R_UPLIMIT = {},
WINDOW_P_STRUCT_UPLIMIT = {},
WINDOW_R_STRUCT_UPLIMIT = {},
VECTOR_REPETITION_CNN = {},
RANDOM_SEED = {},
K_FOLD = {},
BATCH_SIZE = {},
FIRST_TRAIN_EPOCHS = {},
SECOND_TRAIN_EPOCHS = {},
PATIENCES = {},
FIRST_OPTIMIZER = {},
SECOND_OPTIMIZER = {},
SGD_LEARNING_RATE = {},
ADAM_LEARNING_RATE = {},
FREEZE_SUB_MODELS = {},
CODING_FREQUENCY = {},
MONITOR = {},
MIN_DELTA = {},
    """.format(DATA_SET, WINDOW_P_UPLIMIT, WINDOW_R_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT,
               WINDOW_R_STRUCT_UPLIMIT, VECTOR_REPETITION_CNN,
               RANDOM_SEED, K_FOLD, BATCH_SIZE, FIRST_TRAIN_EPOCHS, SECOND_TRAIN_EPOCHS, PATIENCES, FIRST_OPTIMIZER,
               SECOND_OPTIMIZER, SGD_LEARNING_RATE, ADAM_LEARNING_RATE,
               FREEZE_SUB_MODELS, CODING_FREQUENCY, MONITOR, MIN_DELTA)
)

out.write(settings)

P_L = sum_power(7, 1, WINDOW_P_UPLIMIT)  # protein sequence feature dimension
P_S_L = sum_power(7, 1, WINDOW_P_UPLIMIT) + sum_power(4, 1, WINDOW_P_STRUCT_UPLIMIT)  # protein sequence feature + struct feature dimension
P_S_F_L1 = sum_power(7, 1, WINDOW_P_UPLIMIT) + sum_power(4, 1, WINDOW_P_STRUCT_UPLIMIT) + 11 + 80  # protein sequence feature + struct feature + motif feature + physicochemical properties feature dimension
P_S_F_L2 = P_S_F_L1
P_S_F_L3 = P_S_F_L1
R_L = sum_power(4, 1, WINDOW_R_UPLIMIT)  # rna sequence feature dimension
R_S_L = sum_power(4, 1, WINDOW_R_UPLIMIT) + sum_power(7, 1, WINDOW_R_STRUCT_UPLIMIT)  # rna sequence feature + struct feature dimension
R_S_F_L1 = sum_power(4, 1, WINDOW_R_UPLIMIT) + sum_power(7, 1, WINDOW_R_STRUCT_UPLIMIT) + 18 + 20  # rna sequence feature + struct feature + motif feature + physicochemical properties feature dimension
R_S_F_L2 = R_S_F_L1
R_S_F_L3 = R_S_F_L1
# read rna-protein pairs and sequences from data files
pos_pairs, neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs = load_data(DATA_SET)

# sequence encoder instances
PE = ProEncoder(WINDOW_P_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN)
RE = RNAEncoder(WINDOW_R_UPLIMIT, WINDOW_R_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN)


print("Coding positive protein-rna pairs.\n")
samples = coding_pairs(pos_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, PE, RE, kind=1)
positive_sample_number = len(samples)
print("Coding negative protein-rna pairs.\n")
samples += coding_pairs(neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, PE, RE, kind=0)
negative_sample_number = len(samples) - positive_sample_number
sample_num = len(samples)

# positive and negative sample numbers
print('\nPos samples: {}, Neg samples: {}.\n'.format(positive_sample_number, negative_sample_number))
out.write('\nPos samples: {}, Neg samples: {}.\n'.format(positive_sample_number, negative_sample_number))

X, y = pre_process_data(samples=samples)

# K-fold CV processes
print('\n\nK-fold cross validation processes:\n')
out.write('\n\nK-fold cross validation processes:\n')
for fold in range(K_FOLD):
    train = [i for i in range(sample_num) if i%K_FOLD !=fold]
    test = [i for i in range(sample_num) if i%K_FOLD ==fold]

    # generate train and test data
    X_train_conjoint = [X[0][0][train], X[0][1][train]]
    X_train_conjoint_struct = [X[1][0][train], X[1][1][train]]
    X_train_conjoint_struct_fea1 = [X[2][0][train], X[2][1][train]]
    X_train_conjoint_struct_fea2 = [X[3][0][train], X[3][1][train]]
    X_train_conjoint_struct_fea3 = [X[4][0][train], X[4][1][train]]
    X_train_conjoint_cnn = [X[5][0][train], X[5][1][train]]
    X_train_conjoint_struct_cnn = [X[6][0][train], X[6][1][train]]
    X_train_conjoint_struct_fea1_cnn = [X[7][0][train], X[7][1][train]]
    X_train_conjoint_struct_fea2_cnn = [X[8][0][train], X[8][1][train]]
    X_train_conjoint_struct_fea3_cnn = [X[9][0][train], X[9][1][train]]
    X_train_conjoint_previous = [X[10][0][train], X[10][1][train]]

    X_test_conjoint = [X[0][0][test], X[0][1][test]]
    X_test_conjoint_struct = [X[1][0][test], X[1][1][test]]
    X_test_conjoint_struct_fea1 = [X[2][0][test], X[2][1][test]]
    X_test_conjoint_struct_fea2 = [X[3][0][test], X[3][1][test]]
    X_test_conjoint_struct_fea3 = [X[4][0][test], X[4][1][test]]
    X_test_conjoint_cnn = [X[5][0][test], X[5][1][test]]
    X_test_conjoint_struct_cnn = [X[6][0][test], X[6][1][test]]
    X_test_conjoint_struct_fea1_cnn = [X[7][0][test], X[7][1][test]]
    X_test_conjoint_struct_fea2_cnn = [X[8][0][test], X[8][1][test]]
    X_test_conjoint_struct_fea3_cnn = [X[9][0][test], X[9][1][test]]
    X_test_conjoint_previous = [X[10][0][test], X[10][1][test]]

    y_train_mono = y[train]
    y_train = np_utils.to_categorical(y_train_mono, 2)
    y_test_mono = y[test]
    y_test = np_utils.to_categorical(y_test_mono, 2)


    print(str('\n# ' + '=' * 10 + " Fold {} " + "=" * 10 + '\n').format(fold))
    out.write(str('\n# ' + '=' * 10 + " Fold {} " + "=" * 10 + '\n').format(fold))
    model_metrics = {'RPI-EDLCN': np.zeros(7)}

    model_weight_path = result_save_path + 'weights.hdf5'
    module_index = 0

    # RPI-EDLCN module

    stage = 'RPI-EDLCN'
    print("\n# Module RPI-EDLCN #\n")
    x = X_train_conjoint_struct_fea1_cnn + X_train_conjoint_struct_fea2_cnn + X_train_conjoint_struct_fea3
    x_te = X_test_conjoint_struct_fea1_cnn + X_test_conjoint_struct_fea2_cnn + X_test_conjoint_struct_fea3
    # create model
    encoders_pro, encoders_rna = get_auto_encoders(X_train_conjoint_struct_fea3, X_test_conjoint_struct_fea3)
    model_RPI_EDLCN = RPI_EDLCN(encoders_pro, encoders_rna, P_S_F_L1, R_S_F_L1, P_S_F_L2, R_S_F_L2, P_S_F_L3, R_S_F_L3, VECTOR_REPETITION_CNN)
    callbacks = get_callback_list(PATIENCES[0], result_save_path, stage, fold, x_te, y_test)

    # first train
    model_RPI_EDLCN.compile(loss='categorical_crossentropy', optimizer=get_optimizer(FIRST_OPTIMIZER),
                                    metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = False
    model_RPI_EDLCN.fit(x=X_train_conjoint_struct_fea1_cnn + X_train_conjoint_struct_fea2_cnn + X_train_conjoint_struct_fea3,
                                y=y_train,
                                epochs=FIRST_TRAIN_EPOCHS[0],
                                batch_size=BATCH_SIZE,
                                verbose=VERBOSE,
                                shuffle=SHUFFLE,
                                callbacks=[callbacks[0]])

    # second train
    model_RPI_EDLCN.compile(loss='categorical_crossentropy', optimizer=get_optimizer(SECOND_OPTIMIZER),
                                    metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = True
    model_RPI_EDLCN.fit(x=X_train_conjoint_struct_fea1_cnn + X_train_conjoint_struct_fea2_cnn + X_train_conjoint_struct_fea3,
                                y=y_train,
                                epochs=SECOND_TRAIN_EPOCHS[0],
                                batch_size=BATCH_SIZE,
                                verbose=VERBOSE,
                                shuffle=SHUFFLE,
                                callbacks=callbacks)

    # model_RPI_EDLCN.save('model_RPI_EDLCN.h5')

    # test
    y_test_predict = model_RPI_EDLCN.predict(x_te)
    model_metrics['RPI-EDLCN'] = np.array(calc_metrics(y_test[:, 1], y_test_predict[:, 1]))
    print('Best performance for module RPI-EDLCN:\n'
          + 'ACC = ' + str(model_metrics['RPI-EDLCN'][0]) + ' ' + 'SN = ' + str(
        model_metrics['RPI-EDLCN'][1]) + ' '
          + 'SP = ' + str(model_metrics['RPI-EDLCN'][2]) + ' ' + 'PRE = ' + str(
        model_metrics['RPI-EDLCN'][3]) + ' '
          + 'F1_measure = ' + str(model_metrics['RPI-EDLCN'][4]) + ' '
          + 'MCC = ' + str(model_metrics['RPI-EDLCN'][5]) + ' ' + 'AUC = ' + str(
        model_metrics['RPI-EDLCN'][6]) + '\n')

    # =================================================================

    for key in model_metrics:
        out.write(key + " : " +  'ACC = ' + str(model_metrics[key][0]) + ' ' + 'SN = ' + str(model_metrics[key][1]) + ' '
              + 'SP = ' + str(model_metrics[key][2]) + ' ' + 'PRE = ' + str(model_metrics[key][3]) + ' '
              + 'F1_measure = ' + str(model_metrics[key][4]) + ' '
              + 'MCC = ' + str(model_metrics[key][5]) + ' ' + 'AUC = ' + str(model_metrics[key][6]) + '\n')

    for key in model_metrics:
        metrics_whole[key] += model_metrics[key]


for key in metrics_whole.keys():
    metrics_whole[key] /= K_FOLD
    print('\nMean metrics in {} fold:\n'.format(K_FOLD) + key + " : "
          + 'ACC = ' + str(metrics_whole[key][0]) + ' ' + 'SN = ' + str(metrics_whole[key][1]) + ' '
          + 'SP = ' + str(metrics_whole[key][2]) + ' ' + 'PRE = ' + str(metrics_whole[key][3]) + ' '
          + 'F1_measure = ' + str(metrics_whole[key][4]) + ' ' + 'MCC = ' + str(metrics_whole[key][5]) + ' '
          + 'AUC = ' + str(metrics_whole[key][6]) + '\n')
    out.write('\nMean metrics in {} fold:\n'.format(K_FOLD) + key + " : " + 'ACC = ' + str(metrics_whole[key][0]) + ' ' + 'SN = ' + str(metrics_whole[key][1]) + ' '
              + 'SP = ' + str(metrics_whole[key][2]) + ' ' + 'PRE = ' + str(metrics_whole[key][3]) + ' '
              + 'F1_measure = ' + str(metrics_whole[key][4]) + ' '
              + 'MCC = ' + str(metrics_whole[key][5]) + ' ' + 'AUC = ' + str(metrics_whole[key][6]) + '\n')
out.flush()
out.close()
