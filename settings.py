"""
settings.py script contains different parameters that specify model and training procedure.
experiment_no is used for training different experiments and it should be incremented before training.
"""
import os
import random
import numpy as np
import tensorflow as tf
import shutil


########################################################################################################################
# TRAINING SETTINGS
########################################################################################################################

experiment_no = 1

batch_size = 16

image_size = 512
stretch = False

balance_dataset = True #False

########################################################################################################################

#-----------------------------------------------------------------------------------------------------------------------
# DEFAULT TRAINING DISTRIBUTION
#-----------------------------------------------------------------------------------------------------------------------

# train (T) 50%, val (V) 20%, test (S) 50%
#subset_distribution = [{'filter': [], 'distribution': ['T', 'S', 'T', 'T', 'V', 'S', 'T', 'T', 'S', 'V']}]
#data_sub_folder = ''

#-----------------------------------------------------------------------------------------------------------------------
# DATASET SIZE EXPERIMENTS
#-----------------------------------------------------------------------------------------------------------------------

# train (T) 10%, val (V) 10%, test (S) 10%
subset_distribution = [{'filter': [], 'distribution': ['T', '-', '-', '-', 'V', 'S', '-', '-', '-', '-']}]
data_sub_folder = 'train10-val10-test10'

# train (T) 20%, val (V) 10%, test (S) 10%
#subset_distribution = [{'filter': [], 'distribution': ['T', '-', '-', '-', 'V', 'S', 'T', '-', '-', '-']}]
#data_sub_folder = 'train20-val10-test10'

# train (T) 30%, val (V) 10%, test (S) 10%
#subset_distribution = [{'filter': [], 'distribution': ['T', '-', 'T', '-', 'V', 'S', 'T', '-', '-', '-']}]
#data_sub_folder = 'train30-val10-test10'

# train (T) 40%, val (V) 10%, test (S) 10%
#subset_distribution = [{'filter': [], 'distribution': ['T', '-', 'T', '-', 'V', 'S', 'T', '-', 'T', '-']}]
#data_sub_folder = 'train40-val10-test10'

# train (T) 50%, val (V) 10%, test (S) 10%
#subset_distribution = [{'filter': [], 'distribution': ['T', 'T', 'T', '-', 'V', 'S', 'T', '-', 'T', '-']}]
#data_sub_folder = 'train50-val10-test10'

# train (T) 60%, val (V) 10%, test (S) 10%
#subset_distribution = [{'filter': [], 'distribution': ['T', 'T', 'T', '-', 'V', 'S', 'T', 'T', 'T', '-']}]
#data_sub_folder = 'train60-val10-test10'

# train (T) 70%, val (V) 10%, test (S) 10%
#subset_distribution = [{'filter': [], 'distribution': ['T', 'T', 'T', 'T', 'V', 'S', 'T', 'T', 'T', '-']}]
#data_sub_folder = 'train70-val10-test10'

# train (T) 80%, val (V) 10%, test (S) 10%
#subset_distribution = [{'filter': [], 'distribution': ['T', 'T', 'T', 'T', 'V', 'S', 'T', 'T', 'T', 'T']}]
#data_sub_folder = 'train80-val10-test10'

#-----------------------------------------------------------------------------------------------------------------------
# DATASET BALANCE EXPERIMENTS
#-----------------------------------------------------------------------------------------------------------------------

# BALANCE TEST 1
"""
subset_distribution = [
    {
        'filter': ['105', '106', '111', '113', '119', '120', '128', '134', '146', '161', '193'],
        'distribution': ['T', 'T', 'T', '-', 'V', 'S', 'T', 'T', 'T', '-']  # train (T) 60%, val (V) 10%, test (S) 10%
    },
    {
        'filter': [],
        'distribution': ['T', 'T', 'T', 'T', 'V', 'S', 'T', 'T', 'T', 'T']  # train (T) 80%, val (V) 10%, test (S) 10%
    }
]
data_sub_folder = "bt1-22v80-11v60"
"""

# BALANCE TEST 2
"""
subset_distribution = [
    {
        'filter': ['105', '106', '111', '113', '119', '120', '128', '134', '146', '161', '193'],
        'distribution': ['T', '-', 'T', '-', 'V', 'S', 'T', '-', 'T', '-']  # train (T) 40%, val (V) 10%, test (S) 10%
    },
    {
        'filter': [],
        'distribution': ['T', 'T', 'T', 'T', 'V', 'S', 'T', 'T', 'T', 'T']  # train (T) 80%, val (V) 10%, test (S) 10%
    }
]
data_sub_folder = "bt2-22v80-11v40"
"""

# BALANCE TEST 3
"""
subset_distribution = [
    {
        'filter': ['105', '106', '111', '113', '119', '120', '128', '134', '146', '161', '193'],
        'distribution': ['T', '-', '-', '-', 'V', 'S', 'T', '-', '-', '-']  # train (T) 20%, val (V) 10%, test (S) 10%
    },
    {
        'filter': [],
        'distribution': ['T', 'T', 'T', 'T', 'V', 'S', 'T', 'T', 'T', 'T']  # train (T) 80%, val (V) 10%, test (S) 10%
    }
]
data_sub_folder = "bt3-22v80-11v20"
"""

# BALANCE TEST 4
"""
subset_distribution = [
    {
        'filter': ['101', '103', '109', '111', '113', '122', '125', '136', '156', '182'],
        'distribution': ['T', 'T', 'T', '-', 'V', 'S', 'T', 'T', 'T', '-']  # train (T) 60%, val (V) 10%, test (S) 10%
    },
    {
        'filter': ['105', '106', '112', '113', '119', '120', '134', '146', '161', '193'],
        'distribution': ['T', '-', 'T', '-', 'V', 'S', 'T', '-', 'T', '-']  # train (T) 40%, val (V) 10%, test (S) 10%
    },
    {
        'filter': [],
        'distribution': ['T', 'T', 'T', 'T', 'V', 'S', 'T', 'T', 'T', 'T']  # train (T) 80%, val (V) 10%, test (S) 10%
    }
]
data_sub_folder = "bt4-13v80-10v60-10v40"
"""

# BALANCE TEST 5
"""
subset_distribution = [
    {
        'filter': ['101', '103', '109', '111', '113', '122', '125', '136', '156', '182'],
        'distribution': ['T', 'T', 'T', '-', 'V', 'S', 'T', 'T', 'T', '-']  # train (T) 60%, val (V) 10%, test (S) 10%
    },
    {
        'filter': ['105', '106', '112', '113', '119', '120', '134', '146', '161', '193'],
        'distribution': ['T', '-', '-', '-', 'V', 'S', 'T', '-', '-', '-']  # train (T) 20%, val (V) 10%, test (S) 10%
    },
    {
        'filter': [],
        'distribution': ['T', 'T', 'T', 'T', 'V', 'S', 'T', 'T', 'T', 'T']  # train (T) 80%, val (V) 10%, test (S) 10%
    }
]
data_sub_folder = "bt5-13v80-10v60-10v20"
"""

# BALANCE TEST 6
"""
subset_distribution = [
    {
        'filter': ['101', '103', '109', '111', '113', '122', '125', '136', '156', '182'],
        'distribution': ['T', 'T', 'T', '-', 'V', 'S', 'T', 'T', 'T', '-']  # train (T) 60%, val (V) 10%, test (S) 10%
    },
    {
        'filter': ['105', '106', '112', '113', '119', '120', '134', '146', '161', '193'],
        'distribution': ['T', '-', '-', '-', 'V', 'S', '-', '-', '-', '-']  # train (T) 10%, val (V) 10%, test (S) 10%
    },
    {
        'filter': [],
        'distribution': ['T', 'T', 'T', 'T', 'V', 'S', 'T', 'T', 'T', 'T']  # train (T) 80%, val (V) 10%, test (S) 10%
    }
]
data_sub_folder = "bt6-13v80-10v60-10v10"
"""

# BALANCE TEST 7
"""
subset_distribution = [
    {
        'filter': ['101', '103', '109', '111', '113', '122', '125', '136', '156', '182'],
        'distribution': ['T', '-', 'T', '-', 'V', 'S', 'T', '-', 'T', '-']  # train (T) 40%, val (V) 10%, test (S) 10%
    },
    {
        'filter': ['105', '106', '112', '113', '119', '120', '134', '146', '161', '193'],
        'distribution': ['T', '-', '-', '-', 'V', 'S', 'T', '-', '-', '-']  # train (T) 20%, val (V) 10%, test (S) 10%
    },
    {
        'filter': [],
        'distribution': ['T', 'T', 'T', 'T', 'V', 'S', 'T', 'T', 'T', 'T']  # train (T) 80%, val (V) 10%, test (S) 10%
    }
]
data_sub_folder = "bt7-13v80-10v40-10v20"
"""

# BALANCE TEST 8
"""
subset_distribution = [
    {
        'filter': ['101', '103', '109', '111', '113', '122', '125', '136', '156', '182'],
        'distribution': ['T', '-', '-', '-', 'V', 'S', 'T', '-', '-', '-']  # train (T) 20%, val (V) 10%, test (S) 10%
    },
    {
        'filter': ['105', '106', '112', '113', '119', '120', '134', '146', '161', '193'],
        'distribution': ['T', '-', '-', '-', 'V', 'S', '-', '-', '-', '-']  # train (T) 10%, val (V) 10%, test (S) 10%
    },
    {
        'filter': [],
        'distribution': ['T', 'T', 'T', 'T', 'V', 'S', 'T', 'T', 'T', 'T']  # train (T) 80%, val (V) 10%, test (S) 10%
    }
]
data_sub_folder = "bt8-13v80-10v20-10v10"
"""

########################################################################################################################

#optimizer_name = 'SGD'
#optimizer_name = 'RMSprop'
optimizer_name = 'Adam'

if optimizer_name == 'SGD':
    init_lr = 1e-2
    optimizer = tf.keras.optimizers.SGD(learning_rate=init_lr, momentum=0.9)
elif optimizer_name == 'RMSprop':
    init_lr = 1e-3
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=init_lr)
elif optimizer_name == 'Adam':
    init_lr = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)

data_augmentation = True
cosine_annealing = False #True

monitor_loss = False #True #False
if monitor_loss:
    val_monitor = ('val_loss', 'min')
else:
    val_monitor = ('val_sparse_categorical_accuracy', 'max')

lr_scale = 0.01
lr_period = 10
lr_decay = 0.7
if cosine_annealing:
    epochs = 100 * lr_period
    epochs_warmup = lr_period
    reduce_lr_patience = lr_period
    early_stopping_patience = 3 * lr_period
else:
    epochs = 1000
    epochs_warmup = 10
    reduce_lr_patience = 10
    early_stopping_patience = 30

join_test_with_train = False

heatmaps_for_test_images_only = True #False


########################################################################################################################
# MODEL SETTINGS
########################################################################################################################

dropout_rate = 0
hidden_neurons = 0

#architecture = 'ResNet50'
#architecture = 'ResNet50V2'
#architecture = 'EfficientNetB0'
#architecture = 'EfficientNetB1'
architecture = 'EfficientNetB2'
#architecture = 'EfficientNetB3'
#architecture = 'EfficientNetB4'
#architecture = 'EfficientNetB5'
#architecture = 'EfficientNetB6'


########################################################################################################################
# FOLDER SETTINGS
########################################################################################################################

root_folder = os.path.join('D:\\', 'Datasets', 'AIAQUAMI-EPT33')

data_folder = os.path.join(root_folder, 'data')
original_data_folder = os.path.join(data_folder, 'data_original')
if data_sub_folder != '':
    data_folder = os.path.join(root_folder, 'data', 'data_' + data_sub_folder)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

tmp_folder = os.path.join(root_folder, 'tmp')
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)
if data_sub_folder != '':
    tmp_folder = os.path.join(root_folder, 'tmp', 'results_' + data_sub_folder)
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

train_folder = os.path.join(data_folder, 'train_{}{}'.format(image_size, '_st' if stretch else ''))
if not os.path.exists(train_folder):
    os.makedirs(train_folder)

val_folder = os.path.join(data_folder, 'val_{}{}'.format(image_size, '_st' if stretch else ''))
if not os.path.exists(val_folder):
    os.makedirs(val_folder)

test_folder = os.path.join(data_folder, 'test_{}{}'.format(image_size, '_st' if stretch else ''))
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

curr_folder_name = '{}_{}{}px_{}_do{}_hn{}_da{}_ca{}_lr{}_{}_{}_bs{}_bl{}'.format('{:03d}'.format(experiment_no),
                                                                                  'st' if stretch else '',
                                                                                  image_size,
                                                                                  architecture,
                                                                                  dropout_rate,
                                                                                  hidden_neurons,
                                                                                  'Y' if data_augmentation else 'N',
                                                                                  'Y' if cosine_annealing else 'N',
                                                                                  init_lr,
                                                                                  optimizer_name,
                                                                                  'loss' if monitor_loss else 'acc',
                                                                                  batch_size,
                                                                                  'Y' if balance_dataset else 'N')
curr_folder = os.path.join(tmp_folder, curr_folder_name)
if not os.path.exists(curr_folder):
    os.mkdir(curr_folder)

src_settings_file = os.path.join(os.getcwd(), 'settings.py')
dst_settings_file = os.path.join(curr_folder, 'settings.py')
if os.path.exists(dst_settings_file):
    os.remove(dst_settings_file)
shutil.copyfile(src_settings_file, dst_settings_file)

experiments_file = os.path.join(tmp_folder, 'experiments.csv')
if not os.path.exists(experiments_file):
    with open(experiments_file, 'w') as f:
        f.write('ExperimentNo,ImageSize,Architecture,Dropout,HiddenNeurons,DataAugmentation,CosineAnnealing,'
                'InitLearningRate,Optimizer,MonitorLoss,BatchSize,TrainAcc,ValAcc,TestAcc\r\n')
