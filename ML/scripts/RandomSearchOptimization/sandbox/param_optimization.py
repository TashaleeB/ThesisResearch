from __future__ import print_function, division, absolute_import

# Imports
import json, os, sys, time, math, matplotlib, h5py
import numpy as np

#random seed to control the reproducability
seed = 8675309
np.random.seed(seed)
#np.random.seed(datetime.datetime.now().microsecond)

import tensorflow as tf
#import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

from matplotlib import cm
from matplotlib import gridspec
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter

from tensorflow import keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import backend as K

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

data_path = '/pylon5/as5phnp/tbilling/data/'
reionfilename = data_path+'t21_snapshots_nowedge_v8.hdf5'
project_name='hp_opt'
inputdir = '/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/'+project_name+'/'
inputFile = reionfilename
outputdir = inputdir

steps = 100
factor =1000.
nfold = 10

hp = HyperParameters()

TrainIndex_7 = np.array([0,  1,   2,   3,   4,   7,   9,  11,  12,  13,  14,  15,  17,  19,  20,  22,  23,  24,
                         27,  28,  29,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  42,  43,  44,  45,  46,
                         47,  48,  49,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  66,
                         67,  68,  69,  70,  71,  72,  73,  77,  78,  79,  81,  83,  84,  86,  88,  91,  92,  93,
                         94,  95,  96,  97,  98, 100, 101, 102, 103, 105, 106, 107, 108, 110, 111, 112, 113, 114,
                         115, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129, 131, 132, 134, 135, 136,
                         137, 138, 139, 141, 142, 143, 144, 145, 146, 147, 148, 152, 153, 154, 156, 157, 158, 159,
                         160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 173, 174, 176, 177, 179, 180, 183,
                         184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 197, 198, 199, 200, 203, 204,
                         207, 208, 209, 210, 211, 213, 214, 215, 216, 218, 219, 220, 222, 224, 226, 227, 231, 232,
                         233, 234, 235, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
                         252, 253, 254, 255, 256, 257, 258, 259, 261, 262, 263, 264, 265, 266, 267, 269, 270, 271,
                         272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289,
                         290, 291, 292, 293, 294, 295, 296, 297, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308,
                         310, 311, 312, 313, 314, 315, 317, 319, 320, 321, 322, 323, 324, 325, 326, 328, 329, 330,
                         333, 334, 335, 337, 338, 339, 340, 341, 343, 344, 346, 347, 349, 350, 351, 352, 353, 354,
                         355, 356, 357, 358, 360, 361, 362, 363, 364, 365, 367, 368, 369, 370, 371, 372, 373, 374,
                         375, 376, 377, 380, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395,
                         396, 397, 398, 399, 401, 404, 405, 406, 407, 408, 409, 410, 411, 413, 415, 416, 417, 418,
                         419, 420, 422, 423, 424, 426, 427, 428, 429, 430, 432, 433, 435, 436, 437, 438, 439, 440,
                         441, 442, 443, 444, 445, 446, 447, 449, 452, 453, 457, 458, 459, 460, 461, 462, 463, 465,
                         467, 468, 470, 474, 477, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491,
                         492, 493, 494, 495, 496, 497, 498, 499, 502, 503, 504, 506, 508, 509, 510, 511, 512, 513,
                         514, 515, 516, 517, 518, 519, 520, 522, 523, 524, 526, 527, 529, 530, 531, 532, 533, 534,
                         535, 536, 538, 539, 540, 542, 543, 544, 545, 546, 548, 549, 550, 551, 553, 554, 555, 556,
                         557, 558, 559, 561, 563, 564, 565, 567, 568, 570, 571, 572, 573, 574, 575, 576, 577, 578,
                         581, 582, 583, 584, 585, 587, 588, 589, 592, 593, 595, 596, 597, 599, 600, 601, 602, 604,
                         606, 607, 608, 609, 610, 611, 612, 613, 614, 616, 617, 618, 619, 620, 621, 623, 624, 626,
                         627, 629, 632, 633, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 646, 647, 648, 649,
                         650, 651, 652, 653, 655, 656, 657, 658, 659, 660, 662, 663, 664, 666, 670, 671, 672, 673,
                         675, 677, 678, 680, 682, 683, 684, 685, 686, 689, 690, 691, 692, 693, 695, 696, 697, 698,
                         699, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717,
                         718, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 735, 736, 737, 738,
                         739, 741, 742, 743, 748, 751, 752, 753, 754, 755, 756, 758, 760, 761, 762, 763, 764, 765,
                         766, 767, 768, 771, 772, 773, 774, 775, 776, 779, 781, 782, 783, 784, 785, 786, 787, 789,
                         790, 791, 792, 793, 794, 795, 796, 797, 798, 800, 802, 804, 806, 808, 809, 810, 811, 813,
                         814, 815, 816, 817, 818, 819, 820, 822, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833,
                         834, 835, 837, 838, 839, 840, 842, 843, 844, 845, 848, 849, 850, 851, 852, 853, 854, 855,
                         856, 858, 859, 860, 861, 863, 864, 866, 867, 868, 869, 871, 872, 873, 874, 875, 876, 877,
                         878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 893, 894, 896, 897,
                         899, 901, 902, 903, 904, 905, 909, 910, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921,
                         922, 923, 925, 927, 928, 929, 931, 932, 934, 935, 937, 939, 940, 941, 942, 943, 944, 945,
                         946, 947, 948, 949, 951, 952, 953, 955, 957, 958, 959, 960, 961, 962, 963, 964, 965, 967,
                         968, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 982, 983, 984, 985, 986, 987, 988,
                         989, 990, 991, 992, 993, 996, 997, 998])

TestIndex_7 = np.array([5,   6,   8,  10,  16,  18,  21,  25,  26,  30,  41,  50,  65,  74,  75,  76,  80,  82,
                        85,  87,  89,  90,  99, 104, 109, 116, 127, 130, 133, 140, 149, 150, 151, 155, 163, 172,
                        175, 178, 181, 182, 191, 201, 202, 205, 206, 212, 217, 221, 223, 225, 228, 229, 230, 236,
                        260, 268, 298, 309, 316, 318, 327, 331, 332, 336, 342, 345, 348, 359, 366, 378, 379, 381,
                        400, 402, 403, 412, 414, 421, 425, 431, 434, 448, 450, 451, 454, 455, 456, 464, 466, 469,
                        471, 472, 473, 475, 476, 478, 500, 501, 505, 507, 521, 525, 528, 537, 541, 547, 552, 560,
                        562, 566, 569, 579, 580, 586, 590, 591, 594, 598, 603, 605, 615, 622, 625, 628, 630, 631,
                        634, 645, 654, 661, 665, 667, 668, 669, 674, 676, 679, 681, 687, 688, 694, 700, 719, 733,
                        734, 740, 744, 745, 746, 747, 749, 750, 757, 759, 769, 770, 777, 778, 780, 788, 799, 801,
                        803, 805, 807, 812, 821, 823, 836, 841, 846, 847, 857, 862, 865, 870, 892, 895, 898, 900,
                        906, 907, 908, 911, 924, 926, 930, 933, 936, 938, 950, 954, 956, 966, 969, 970, 981, 994,
                        995, 999])


def readLabels(ind=None, **params):
    """
    read in labels only
    (to use with batches """
    f = h5py.File(inputFile, 'r')

    if ind is None:
        labels = np.asarray(f['Data'][u'snapshot_labels'])  #(N_realizations, N_parameters)
    else:
        labels = np.asarray(f['Data'][u'snapshot_labels'][:, ind])

    if labels.ndim == 1:
        print('training on just one param.')
        print('starting with the following shape, dim:', labels.shape, labels.ndim)
        if labels.ndim > 1:
            labels = labels[:, params['predictoneparam']]
    elif labels.shape[1] == 2:
        print('training on two params.')
        print('starting with the following shape, dim:', labels.shape, labels.ndim)
        if labels.ndim > 1:
            labels = labels[:, ind]

    #if there's only one label per image, we'll have to reshape it:
    if labels.ndim == 1:
        print('reshaping data...')
        labels = labels.reshape(-1, 1)

    return labels

def readImages(ind, **params):
    """ 
    read in data
    """

    print('reading data from', inputFile)

    f = h5py.File(inputFile, 'r')

    #if params['debug'] == True:
    #    data = np.asarray(f['Data'][u't21_snapshots'][ind,:,0:16,0:16])
    if 'crop' in params:
        #print('cropping.')
        #use just the top corner of the images
        data = np.asarray(f['Data'][u't21_snapshots'][ind,:,0:params['crop'],0:params['crop']])
    else:
        #use everything!
        print('reading all data', len(ind))
        data = np.asarray(f['Data'][u't21_snapshots'][ind,:,:,:]) # (N_realizations, N_redshifts, N_pix, N_pix)
        #print('loaded data', len(ind))

    print('finished loading data.', data.shape)

    data  = data.transpose(0,2,3,1) #(N_realizations, N_pix, N_pix, N_redshifts)

    return data, data[0].shape

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

trainlabels = readLabels(ind=None)[TrainIndex_7,5]*factor
trainlabels = trainlabels.reshape(-1, 1)
images,shape =readImages(ind=np.arange(1000))

def build_model(hp):
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',input_shape=images.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-1,1e-2, 1e-3, 1e-4])),
               loss=hp.Choice('loss',values=['mse','mae','mape','msle']),
               metrics=['mse', 'mae', 'mape', r2_keras])
    #plot_model(model, to_file="images/model_plot.png", show_shapes=True, show_layer_names=True)
    print(model.summary())
    return model


def savePreds(model, eval_data, eval_labels, Ntot, fold, istart=0):
    outputFile = os.path.join(outputdir, "results{}_{:d}.npy".format(steps,fold))

    # The Predict() method -  is foer the actual prediction. It generates output predictions for the input samples.
    preds = model.predict(eval_data, verbose=0).flatten() #0 = silent

    Nregressparams = len(eval_labels[0])

    results = np.zeros((Ntot, Nregressparams),
                           dtype = [('truth', 'f'), ('prediction', 'f'), ('fold', 'i')])
    iend = istart+len(eval_labels)

    #print('istart and iend', istart, iend)

    results['fold'][istart:iend] = fold
    results['truth'][istart:iend] = eval_labels
    for n in range(Nregressparams):
        results['prediction'][istart:iend,n] = preds[n::Nregressparams]

    np.save(outputFile, results)

#print start time
os.system("date")

# Start Tuning based on low validation loss
tuner = RandomSearch(build_model,
                     objective='val_loss', # 'loss', 'val_loss', 'val_accuracy'
                     max_trials=20,
                     executions_per_trial=3,
                     directory=outputdir,
                     project_name=project_name)

# Display search overview.
tuner.search_space_summary()

# Performs the hypertuning.
tuner.search(images[TrainIndex_7], trainlabels, epochs=30, validation_split=0.2)

print("\n\n\n")
print("Printing the results of the best model and its hyperparameters with resulting metrics.")
# Show the best models, their hyperparameters, and the resulting metrics.
tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0] # Number of models to show.

# Save best hyperperam Model
print("saving best hyperparameter model ...")
best_model.save(outputdir+"best_hyperparam.h5")

# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')

#print finish time
os.system("date")


def crossvalidate(labels, input_shape, model, data=None, valfrac=0.2, outdir=None, **params):
    #loop through the folds, training on 80% of Nfolds-1, testing on 20% of Nfolds-1,
    #and predicting the remaining fold.
    #valfrac sets the 80/20 split

    params = {'runlabel':'alldata_b_',
            'Nfolds': nfold,
            'debug': False,
            'epochs': steps,
            'crop': 512,
            'predicttwoparams': [1],# [0, 1],
            'patience': 20,
            'learning_rate': 0.1,
            'decay': True}
    
    scores = []
    kf = KFold(n_splits=params['Nfolds'], random_state=seed, shuffle=True)
    kfold_split = np.array(list(kf.split(np.arange(len(labels))))) # [fold,2]
    istart = 0
    for fold, [train_index, test_index] in enumerate(kf.split(np.arange(len(labels)))):
        print('regressing fold', fold+1, 'of', params['Nfolds'], '...')

        print("Train Index: ", train_index, "\n")
        print("Test Index: ", test_index)
        np.savez(outputdir+"train_test_index_{}_{:d}.npz".format(steps,fold),train_index=train_index, test_index=test_index)
        
        log_dir = os.path.join(outputdir, 'output', str(fold))
        cb = keras.callbacks.TensorBoard(log_dir=log_dir,
                                         histogram_freq=10, write_images=True)

        print('number of regression parameters: ', labels.shape[1])

        # The fit() method - Trains the model with the given inputs (and corresponding training labels)
        print('fitting model...')
       
        history = model.fit(data[train_index], labels[train_index], batch_size=32, verbose=2,
                            validation_split=0.2, epochs=steps, callbacks=[cb])

        # Plot training & validation loss values
        plt.figure(figsize=(15,10))
        plt.plot(history.history['loss'][1:],".-")
        plt.plot(history.history['val_loss'][1:],".-")
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.savefig(outputdir+'Loss_Epoch_linearity{}_{:d}.png'.format(steps,fold))
        plt.clf()

        # The evaluate() method - gets the loss statistics on already trained model using the validation (or test) data and the corresponding labels. Returns the loss value and metrics values for the model.
        print('calculating test loss...')
        score = model.evaluate(data[test_index], labels[test_index], batch_size=32, verbose=1) # 1 = progress bar
        # returns: loss
        print('          Test loss:', score)
        print('')
        #scores.append(score)
        
        #save predictions
        #print('saving and plotting predictions for fold', fold+1, '...')
        if fold == 0:
            results = None
        savePreds(model, data[test_index], labels[test_index], len(labels), fold, istart=istart)
        istart += len(test_index)

        # save model
        model_out = os.path.join(outdir, 'fold_{:d}.model'.format(fold))
        print('saving model {}...'.format(model_out))
        model.save(model_out)
        
        n = 0
        results = np.load(outputdir+"results{}_{:d}.npy".format(steps,fold))
        # Plot truth and prediction with target
        plt.figure(figsize=(15,10))
        plt.scatter(results['truth'][:,n]/factor, results['prediction'][:,n]/factor, s=2, lw=0, alpha=0.5, c='k')
        #add a 1-to-1 line:
        x = np.linspace(0.95*np.min(results['truth'][:,n]/factor),
                        1.05*np.max(results['truth'][:,n]/factor), 1000)
        plt.plot(x, x, "r--")
        plt.xlabel('true')
        plt.ylabel('predicted')
        filename = outputdir+"results{}_with_target.png".format(fold)
        plt.savefig(filename)
        plt.clf()
        
        # Save history
        print("Removing Scaling factor ({}) and saving history...".format(factor))
        np.savez(outputdir+"history{}_{:d}".format(steps,fold),
                 loss=np.array(history.history['loss'])/factor,
                 val_loss=np.array(history.history['val_loss'])/factor,
                 r2_keras=np.array(history.history['r2_keras']),
                 val_r2_keras=np.array(history.history['val_r2_keras']),
                 mse=np.array(history.history['mse'])/factor,
                 val_mse=np.array(history.history['val_mse'])/factor,
                 mae=np.array(history.history['mae'])/factor,
                 val_mae=np.array(history.history['val_mae'])/factor,
                 mape=np.array(history.history['mape'])/factor,
                 val_mape=np.array(history.history['val_mape'])/factor)

        # test resetting model
        K.clear_session()

del(trainlabels)
labels = readLabels(ind=None)[:,5]*factor
labels = labels.reshape(-1, 1)

crossvalidate(labels=labels, input_shape=shape, model=best_model, data=images, valfrac=0.2, outdir=outputdir)
