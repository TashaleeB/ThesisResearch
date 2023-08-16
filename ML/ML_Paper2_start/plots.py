"""
This script is responisble for making all of the plots that will go into the second paper "**INSERT TITLE HERE**".
"""

# Import libraries
from __future__ import print_function, division, absolute_import
import os, h5py, glob, io
# Use scikit-learn to grid search the batch size and epochs
import numpy as np, matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import tensorflow as tf

n=0
factor=1000.
h_2 = 0.45321170409999995
low_z_tau = 0.030029479627917934
coefficients = []

"""
# path to test data
data_path_nowedge = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/200steps_v9/" # No wedge filtering
data_path_wedge = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/wedgefilter_v9/"# Wedge Filterd
# load all three different types of data
"""

def single_plot_average_predictions(trainig_data_name, avg_predictions, y_test, model_name, path, filter="nowedge"):
    # make list of predictions using the best model
    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    X_TEST : ndarray
        Input image data. Will be converted to int. Length 1.
    AVG_PREDICTIONS : ndarray
        List of predictions from trained models.
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"


    OUTPUT: Single plot with multiple one-to-one predictions.
    """
    plt.subplots(figsize=(15,10))
    for idx in range(avg_predictions.shape[0]):
        plt.scatter(y_test, avg_predictions[idx], s=20, lw=0, alpha=0.9, label="Model {}".format(str(1+idx)))
    
    plt.plot(y_test,y_test, "k--", linewidth=5, alpha=0.2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    
    plt.savefig(os.path.join(path, filter,
        "single_plot_average_predictions_plots_{}_{}.png".format(trainig_data_name, model_name)))
    plt.savefig(os.path.join(path, filter,
        "single_plot_average_predictions_plots_{}_{}.pdf".format(trainig_data_name, model_name)))
    

def plot_one_to_one_pred(trainig_data_name, model_name, y_test, avg_predictions, path, filter="nowedge"):
    # Plot one-to-one predictions
    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    Y_TEST : ndarray
        Image labels. Will be converted to int. Length 1.
    AVG_PREDICTIONS : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from averaging MC-Sampled models.
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    eg.
        plot_one_to_one_pred(trainig_data_name="ToyModel",
                     y_test=y_test,
                     avg_predictions=np.array([avg_predictions1, avg_predictions2, avg_predictions3, avg_predictions4]),
                     path="name_of_dir/save_here_please/",
                     filter="nowedge")
    """

    
    fig, axes = plt.subplots(figsize=(15,10),nrows=2, ncols=2, sharex=False, sharey=False)
    idx = 0
    for i in range(0,2): # image row
        for j in range(0,2): # image column
            # idx selects the architectures
            #for avg_pred in avg_predictions[idx]: # from that architectures select a model
                # loop through and plot all the predictions for that architectures. It could be any number.

            # plot scatter plot predicted values and one-one line
            axes[i,j].scatter(y_test, avg_predictions, s=20, lw=0, alpha=0.9)#, label="Model {}".format(str(idx)))
            axes[i,j].plot(y_test,y_test, "k--", linewidth=5, alpha=0.2)


            # plot box plot
            axes[i,j].set_ylabel("Predicted", fontsize=16)
            
            axes[i,j].set_xlabel(r"$\tau_{True}$", fontsize=16)

            # set limits on axes
            #axes[i,j].set_xlim(0.0475,0.0675)
            #axes[i,j].set_xlim(0.0475,0.0675)
            #axes[i,j].set_ylim(0.0475,0.0675)
            #axes[i,j].set_ylim(0.0475,0.0675)

            # set scale on axes
            axes[i,j].set_xscale('linear')
            axes[i,j].set_xscale('linear')
            axes[i,j].set_yscale('linear')
            axes[i,j].set_yscale('linear')
            
            plt.setp(axes[i,j].get_yticklabels(), fontsize=16)
            plt.setp(axes[i,j].get_xticklabels(), fontsize=16)

                #axes[0,0].legend(markerscale=2.5)
            idx +=1
        
            #plt.xlabel("Ionized Pixels", size = 16)
            #plt.ylabel("Predictions", size = 16)
            #plt.legend(markerscale=2.5)
    plt.tight_layout()
    plt.savefig(os.path.join(path, filter,
        "one_to_one_mc_sample_avg_plots_{}.png".format(trainig_data_name)))
    plt.savefig(os.path.join(path, filter,
        "one_to_one_mc_sample_avg_plots_{}.pdf".format(trainig_data_name)))
        
    return

def plot_pred_distribution(trainig_data_name, model_name, predictions, y_test, path, filter="nowedge"):
    # data from mc_pred
    
    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    Y_TEST : ndarray
        Image labels. Will be converted to int. Length 1.
    PREDICTIONS : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from MC-Sampled models. I did NOT average the predicted values.
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    eg.
        plot_pred_distribution(trainig_data_name="toymodel", predictions=np.array([predictions1,predictions2,predictions3,predictions4]), y_test=y_test, path="./", filter="nowedge")
    """
    sns.set(font_scale=1.4)
 
    #plt.title("Predicted Number of Ionized {}".format("Pixels",size = 16 ))
    for p in range(predictions.shape[0]): # image row
        fig, axes = plt.subplots(figsize=(15,10),nrows=1, ncols=1, sharex=False, sharey=False)
        sns.distplot(predictions[p,:,0]-y_test[0],label="Predicted Ionized Pixels {}".format(str(y_test[0])))
        sns.distplot(predictions[p,:,1]-y_test[1],label="Predicted Ionized Pixels {}".format(str(y_test[1])))
        sns.distplot(predictions[p,:,2]-y_test[2],label="Predicted Ionized Pixels {}".format(str(y_test[2])))
        sns.distplot(predictions[p,:,3]-y_test[3],label="Predicted Ionized Pixels {}".format(str(y_test[3])))
        sns.distplot(predictions[p,:,4]-y_test[4],label="Predicted Ionized Pixels {}".format(str(y_test[4])))

        plt.setp(axes.get_yticklabels(), fontsize=16)
        plt.setp(axes.get_xticklabels(), fontsize=16)
        plt.legend(markerscale=2.5)
        
        plt.savefig(os.path.join(path, filter,
            "distribution_of_predictions_from_mc_sample_plots_{}_{}_{}.png".format(trainig_data_name, model_name, str(p+1))))
        plt.savefig(os.path.join(path, filter,
            "distribution_of_predictions_from_mc_sample_plots_{}_{}_{}.pdf".format(trainig_data_name, model_name, str(p+1))))
            
    return

def plot_mcMean_vs_DisLambdaMean(trainig_data_name, model_name, yhat, mc_predictions, y_test, path, filter="nowedge"):
    # data from mc_pred
    
    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    Y_TEST : ndarray
        Image labels. Will be converted to int. Length 1.
    YHAT : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from models trained with a DistributionLambda layer at the end.
    MC_PREDICTIONS : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from MC-Sampled models. I did NOT average the predicted values.
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    eg.
        plot_mcMean_vs_DisLambdaMean(trainig_data_name="toymodel",
        yhat=np.array([model(X_test),model(X_test),model(X_test),model(X_test)]),
         mc_predictions=np.array([pred_distLambda1,pred_distLambda2,pred_distLambda3,pred_distLambda4]), y_test=y_test, path="./", filter="nowedge")

    """
 

    plt.figure(figsize=(13,13))
    
    for p in range(mc_predictions.shape[0]):
        plt.plot(yhat[p].mean(), np.mean(mc_predictions[p], axis=0), 'o', label = str(p+1))
    
    plt.plot(y_test,y_test, "k--", linewidth=5)

    plt.xlabel("DistributionLambda Mean", size = 16)
    plt.ylabel("MC-Sample Mean", size = 16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(markerscale=2.5)
    plt.tight_layout()
    
    plt.savefig(os.path.join(path, filter,
        "mean_predictions_from_mc_sample_vs_distlambda_plots_{}_{}.png".format(trainig_data_name, model_name)))
    plt.savefig(os.path.join(path, filter,
        "mean_predictions_from_mc_sample_vs_distlambda_plots_{}_{}.pdf".format(trainig_data_name, model_name)))
    return

def plot_mcSTD_vs_DisLambdaSTD(trainig_data_name, model_name, yhat, mc_predictions, y_test, path, filter_name="nowedge"):
    # data from mc_pred
    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    Y_TEST : ndarray
        Image labels. Will be converted to int. Length 1.
    YHAT : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from models trained with a DistributionLambda layer at the end.
    MC_PREDICTIONS : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from MC-Sampled models. I did NOT average the predicted values.
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    eg.
        plot_mcMean_vs_DisLambdaMean(trainig_data_name="toymodel",
        yhat=np.array([model(X_test),model(X_test),model(X_test),model(X_test)]),
         mc_predictions=np.array([pred_distLambda1,pred_distLambda2,pred_distLambda3,pred_distLambda4]), y_test=y_test, path="./", filter="nowedge")

    """
    plt.figure(figsize=(13,13))
    
    for p in range(mc_predictions.shape[0]):
        plt.plot(yhat[p].stddev(), np.std(mc_predictions[p], axis=0), 'o', label = str(p+1))
    
    plt.plot(y_test,y_test, "k--", linewidth=5)

    plt.xlabel("DistributionLambda Mean", size = 16)
    plt.ylabel("MC-Sample Mean", size = 16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(markerscale=1.5)
    plt.tight_layout()
    
    plt.savefig(os.path.join(path, filter_name,
                             "std_predictions_from_mc_sample_vs_distlambda_plots_{}.png".format(trainig_data_name)))
    plt.savefig(os.path.join(path, filter_name,
                             "std_predictions_from_mc_sample_vs_distlambda_plots_{}.pdf".format(trainig_data_name)))
    
    return
    

def plot_qm_qs_vals(trainig_data_name, model_name, bnn_model, path, filter="nowedge"):

    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    BNN_MODEL : keras.engine.sequential.Sequential
        A single trained Bayesian CNN.
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    eg.
        plot_qm_qs_vals(trainig_data_name="toymodel", bnn_model= model, path="./", filter="nowedge")

    """
    sns.set(font_scale=1.4)
    names = [layer.name for layer in bnn_model.layers if "flipout" in layer.name]
    
    qm_vals = [
        layer.kernel_posterior.mean()
        for layer in bnn_model.layers
        if "flipout" in layer.name
    ]
    qs_vals = [
        layer.kernel_posterior.stddev()
        for layer in bnn_model.layers
        if "flipout" in layer.name
    ]


    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 2, 1)
    for n, qm in zip(names, qm_vals):
        sns.distplot(tf.reshape(qm, shape=[-1]), ax=ax, label=n)
    ax.set_title("Weight means",size = 16 )
    ax.set_xlim([-1.5, 1.5])
    plt.setp(ax.get_yticklabels(), fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=16)
    ax.legend()
    ax = fig.add_subplot(1, 2, 2)
    for n, qs in zip(names, qs_vals):
        sns.distplot(tf.reshape(qs, shape=[-1]), ax=ax)
    ax.set_title("weight stddevs", size = 16)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=16)
    fig.tight_layout()
    #plt.show()
    
    print("saving plots...")
    plt.savefig(os.path.join(path, filter,
        "distribution_of_qm_qs_{}_{}.png".format(trainig_data_name, model_name)))
    plt.savefig(os.path.join(path, filter,
        "distribution_of_qm_qs_{}_{}.pdf".format(trainig_data_name, model_name)))
    
    return

 
def plot_one_to_one_pred_MC_DistributionLambda(trainig_data_name, model_name, yhat, mc_predictions, y_test, path, filter="nowedge"):
    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    Y_TEST : ndarray
        Image labels. Will be converted to int. Length 1.
    YHAT : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from models trained with a DistributionLambda layer at the end.
    MC_PREDICTIONS : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from MC-Sampled models. I did NOT average the predicted values.
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    eg.
        plot_one_to_one_pred_MC_DistributionLambda(trainig_data_name="toymodel",
        yhat=np.array([model(X_test),model(X_test),model(X_test),model(X_test)]),
         mc_predictions=np.array([pred_distLambda1,pred_distLambda2,pred_distLambda3,pred_distLambda4]), y_test=y_test, path="./", filter="nowedge")

    """
    
    fig, axes = plt.subplots(figsize=(15,10),nrows=2, ncols=2, sharex=False, sharey=False)
    # plt.title("Monte Carlo vs Sigma in output layer", size = 16)
    idx = 0
    for i in range(0,2): # image row
        for j in range(0,2): # image column
            # idx selects the architectures

            # plot scatter plot predicted values and one-one line
            axes[i,j].plot(y_test, np.mean(mc_predictions[idx], axis=0), 'r o', alpha = 0.5, label = "Method 1: Monte Carlo Sample")
            axes[i,j].plot(y_test, yhat[idx].mean(), 'c *', alpha = 0.5, label = "Method 2: DistributionLambda")
            axes[i,j].plot(y_test,y_test, "k--", linewidth=5, alpha=0.2)


            # plot box plot
            axes[i,j].set_ylabel("Predicted", fontsize=16)
            
            axes[i,j].set_xlabel(r"$\tau_{True}$", fontsize=16)

            # set limits on axes
            #axes[i,j].set_xlim(0.0475,0.0675)
            #axes[i,j].set_xlim(0.0475,0.0675)
            #axes[i,j].set_ylim(0.0475,0.0675)
            #axes[i,j].set_ylim(0.0475,0.0675)

            # set scale on axes
            axes[i,j].set_xscale('linear')
            axes[i,j].set_xscale('linear')
            axes[i,j].set_yscale('linear')
            axes[i,j].set_yscale('linear')
            
            plt.setp(axes[i,j].get_yticklabels(), fontsize=16)
            plt.setp(axes[i,j].get_xticklabels(), fontsize=16)

            #axes[0,0].legend(markerscale=2.5)
            idx +=1

    plt.legend(markerscale=1.5)
    plt.tight_layout()
    
    # save plots as png and pdf
    plt.savefig(os.path.join(path, filter,
        "plot_one_to_one_pred_MC_DistributionLambda_{}.png".format(trainig_data_name)))
    plt.savefig(os.path.join(path, filter,
        "plot_one_to_one_pred_MC_DistributionLambda_{}.pdf".format(trainig_data_name)))
    
    return
    
def plot_deterministic_model_error1(trainig_data_name, model_name, mc_predictions, y_test, path, filter="nowedge"):

    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    Y_TEST : ndarray
        Image labels. Will be converted to int. Length 1.
    YHAT : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from models trained with a DistributionLambda layer at the end.
    MC_PREDICTIONS : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from MC-Sampled models. I did NOT average the predicted values.
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    eg.
        plot_deterministic_model_error1(trainig_data_name="toymodel",
         mc_predictions=np.array([pred_distribution1,pred_distribution2,pred_distribution3,pred_distribution4]), y_test=y_test, path="./", filter="nowedge")

    """
    
    fig, axes = plt.subplots(figsize=(15,10),nrows=2, ncols=2, sharex=False, sharey=False)
    # plt.title("Monte Carlo vs Sigma in output layer", size = 16)
    idx = 0
    for i in range(0,2): # image row
        for j in range(0,2): # image column
            # idx selects the architectures

            # plot scatter plot predicted values and one-one line
            #axes[i,j].errorbar(y_test[0::20], np.mean(mc_predictions[idx], axis=0)[0::20], yerr=np.std(mc_predictions[idx], axis=0)[0::20], marker='.',
            #             mfc='orangered', mec='green', c='green', mew=2.5, ms=20, ls='', alpha = 0.9, #label="Method 1: Monte Carlo Sample"
            #             )
            axes[i,j].errorbar(y_test[:], np.mean(mc_predictions[idx], axis=0)[:], yerr=np.std(mc_predictions[idx], axis=0)[:], marker='.',
                         mfc='orangered', mec='green', c='green', mew=2.5, ms=20, ls='', alpha = 0.9, #label="Method 1: Monte Carlo Sample"
                         )
            
            #axes[i,j].plot(y_test[0::20],np.mean(mc_predictions[idx], axis=0)[0::20]+np.std(mc_predictions[idx], axis=0)[0::20], 'g-', alpha=0.5,
            #         label='mean + 1 stddev'
            #        )
            """
            axes[i,j].plot(y_test[:],np.mean(mc_predictions[idx], axis=0)[:]+np.std(mc_predictions[idx], axis=0)[:], 'g-', alpha=0.5,
                     label='mean + 1 stddev'
                    )
            """
                    
            #axes[i,j].plot(y_test[0::20],np.mean(mc_predictions[idx], axis=0)[0::20]-np.std(mc_predictions[idx], axis=0)[0::20], 'g-', alpha=0.5,
            #         label='mean - 1 stddev'
            #        )
            """
            axes[i,j].plot(y_test[:],np.mean(mc_predictions[idx], axis=0)[:]-np.std(mc_predictions[idx], axis=0)[:], 'g-', alpha=0.5,
                     label='mean - 1 stddev'
                    )
            """
            
            axes[i,j].plot(y_test,y_test, "k--", linewidth=7, alpha=1.0)

            
            # plot box plot
            axes[i,j].set_ylabel("Predicted", fontsize=16)

            axes[i,j].set_xlabel(r"$\tau_{True}$", fontsize=16)

            # set limits on axes
            #axes[i,j].set_xlim(0.0475,0.0675)
            #axes[i,j].set_xlim(0.0475,0.0675)
            #axes[i,j].set_ylim(0.0475,0.0675)
            #axes[i,j].set_ylim(0.0475,0.0675)

            # set scale on axes
            axes[i,j].set_xscale('linear')
            axes[i,j].set_xscale('linear')
            axes[i,j].set_yscale('linear')
            axes[i,j].set_yscale('linear')
            
            plt.setp(axes[i,j].get_yticklabels(), fontsize=16)
            plt.setp(axes[i,j].get_xticklabels(), fontsize=16)

            #axes[0,0].legend(markerscale=2.5)
            idx +=1

    plt.legend(markerscale=1.5, fontsize='large')
    plt.tight_layout()
    #plt.show()
    
    # save plots as png and pdf
    plt.savefig(os.path.join(path, filter,
        "plot_deterministic_model_error1_{}_{}.png".format(trainig_data_name, model_name)))
    plt.savefig(os.path.join(path, filter,
        "plot_deterministic_model_error1_{}_{}.pdf".format(trainig_data_name, model_name)))
        
    return

def plot_error1(trainig_data_name, model_name, yhat, mc_predictions, y_test, path, filter="nowedge"):

    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    Y_TEST : ndarray
        Image labels. Will be converted to int. Length 1.
    YHAT : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from models trained with a DistributionLambda layer at the end.
    MC_PREDICTIONS : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from MC-Sampled models. I did NOT average the predicted values.
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    eg.
        plot_error1(trainig_data_name="toymodel",
        yhat=np.array([model(X_test),model(X_test),model(X_test),model(X_test)]),
         mc_predictions=np.array([pred_distLambda1,pred_distLambda2,pred_distLambda3,pred_distLambda4]), y_test=y_test, path="./", filter="nowedge")

    """
    
    fig, axes = plt.subplots(figsize=(15,10),nrows=2, ncols=2, sharex=False, sharey=False)
    # plt.title("Monte Carlo vs Sigma in output layer", size = 16)
    idx = 0
    for i in range(0,2): # image row
        for j in range(0,2): # image column
            # idx selects the architectures

            """
            # plot scatter plot predicted values and one-one line
            axes[i,j].errorbar(y_test[0::20], np.mean(mc_predictions[idx], axis=0)[0::20], yerr=np.std(mc_predictions[idx], axis=0)[0::20], marker='.',
                         mfc='orangered', mec='green', c='green', mew=2.5, ms=20, ls='', alpha = 0.9, label="Method 1: Monte Carlo Sample")
            
            axes[i,j].errorbar(y_test[0::20], yhat[idx].mean().numpy()[0::20,0], yerr=yhat[idx].stddev().numpy()[0::20,0], marker='.',
                         mfc='lightpink', mec='blue', c='blue', mew=2.5, ms=20, ls='', alpha = 0.4, label="Method 2: DistributionLambda")

            axes[i,j].plot(y_test[0::20],np.mean(mc_predictions[idx], axis=0)[0::20]+np.std(mc_predictions[idx], axis=0)[0::20], 'g-', alpha=0.5,
                     label='mean + 1 stddev'
                    )
            axes[i,j].plot(y_test[0::20],np.mean(mc_predictions[idx], axis=0)[0::20]-np.std(mc_predictions[idx], axis=0)[0::20], 'g-', alpha=0.5,
                     label='mean - 1 stddev'
                    )

            axes[i,j].plot(y_test[0::20],yhat[idx].mean().numpy()[0::20]+yhat[idx].stddev().numpy()[0::20], '-',color='midnightblue', alpha=0.25,
                     label='mean + 1 stddev'
                    )
            axes[i,j].plot(y_test[0::20],yhat[idx].mean().numpy()[0::20]-yhat[idx].stddev().numpy()[0::20], '-',color='midnightblue', alpha=0.25,
                     label='mean - 1 stddev'
                    )
            axes[i,j].plot(y_test,y_test, "k--", linewidth=7, alpha=1.0)
            """
                        # plot scatter plot predicted values and one-one line
            axes[i,j].errorbar(y_test[:], np.mean(mc_predictions[idx], axis=0)[:], yerr=np.std(mc_predictions[idx], axis=0)[:], marker='.',
                         mfc='orangered', mec='green', c='green', mew=2.5, ms=20, ls='', alpha = 0.9, label="Method 1: Monte Carlo Sample")
            
            axes[i,j].errorbar(y_test[:], yhat[idx].mean().numpy()[:,0], yerr=yhat[idx].stddev().numpy()[:,0], marker='.',
                         mfc='lightpink', mec='blue', c='blue', mew=2.5, ms=20, ls='', alpha = 0.4, label="Method 2: DistributionLambda")
            """
            axes[i,j].plot(y_test[:],np.mean(mc_predictions[idx], axis=0)[:]+np.std(mc_predictions[idx], axis=0)[:], 'g-', alpha=0.5,
                     label='mean + 1 stddev'
                    )
            axes[i,j].plot(y_test[:],np.mean(mc_predictions[idx], axis=0)[:]-np.std(mc_predictions[idx], axis=0)[:], 'g-', alpha=0.5,
                     label='mean - 1 stddev'
                    )

            axes[i,j].plot(y_test[:],yhat[idx].mean().numpy()[:]+yhat[idx].stddev().numpy()[:], '-',color='midnightblue', alpha=0.25,
                     label='mean + 1 stddev'
                    )
            axes[i,j].plot(y_test[:],yhat[idx].mean().numpy()[:]-yhat[idx].stddev().numpy()[:], '-',color='midnightblue', alpha=0.25,
                     label='mean - 1 stddev'
                    )
            """
            axes[i,j].plot(y_test,y_test, "k--", linewidth=7, alpha=1.0)


            # plot box plot
            axes[i,j].set_ylabel("Predicted", fontsize=16)

            axes[i,j].set_xlabel(r"$\tau_{True}$", fontsize=16)

            # set limits on axes
            #axes[i,j].set_xlim(0.0475,0.0675)
            #axes[i,j].set_xlim(0.0475,0.0675)
            #axes[i,j].set_ylim(0.0475,0.0675)
            #axes[i,j].set_ylim(0.0475,0.0675)

            # set scale on axes
            axes[i,j].set_xscale('linear')
            axes[i,j].set_xscale('linear')
            axes[i,j].set_yscale('linear')
            axes[i,j].set_yscale('linear')
            
            plt.setp(axes[i,j].get_yticklabels(), fontsize=16)
            plt.setp(axes[i,j].get_xticklabels(), fontsize=16)

            #axes[0,0].legend(markerscale=2.5)
            idx +=1

    plt.legend(markerscale=1.5, fontsize='large')
    plt.tight_layout()

    
    # save plots as png and pdf
    plt.savefig(os.path.join(path, filter,
        "plot_error1_{}_{}.png".format(trainig_data_name, model_name)))
    plt.savefig(os.path.join(path, filter,
        "plot_error1_{}_{}.pdf".format(trainig_data_name, model_name)))
        
    return
    
    
def plot_deterministic_model_error2(trainig_data_name, model_name, mc_predictions, y_test, path, filter="nowedge"):

    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    Y_TEST : ndarray
        Image labels. Will be converted to int. Length 1.
    YHAT : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from models trained with a DistributionLambda layer at the end.
    MC_PREDICTIONS : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from MC-Sampled models. I did NOT average the predicted values.
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    eg.
        plot_error2(trainig_data_name="toymodel",
        yhat=np.array([model(X_test),model(X_test),model(X_test),model(X_test)]),
         mc_predictions=np.array([pred_distLambda1,pred_distLambda2,pred_distLambda3,pred_distLambda4]), y_test=y_test, path="./", filter="nowedge")

    """
    
    fig, axes = plt.subplots(figsize=(15,10),nrows=2, ncols=2, sharex=False, sharey=False)
    # plt.title("Monte Carlo vs Sigma in output layer", size = 16)
    idx = 0
    for i in range(0,2): # image row
        for j in range(0,2): # image column
            # idx selects the architectures
            
            #
            axes[i,j].yaxis.tick_left()
            axes[i,j].yaxis.set_label_position("right")

            #axes[i,j].yaxis.set_minor_locator(AutoMinorLocator())
            axes[i,j].tick_params(which='both', width=2)
            axes[i,j].tick_params(which='major', length=7)
            axes[i,j].tick_params(which='minor', length=4, color='r')

            # plot error
            axes[i,j].errorbar(y_test[:], (np.mean(mc_predictions[idx], axis=0)-y_test)[:], yerr=np.std(mc_predictions[idx], axis=0)[:], marker='.', mfc='red', mec='green', c='green', mew=2.5, ms=20, ls='', label="Method 1: Monte Carlo Sample")

            #axes[i,j].errorbar(y_test[0::10], (np.mean(mc_predictions[idx], axis=0)-y_test)[0::10], yerr=np.std(mc_predictions[idx], axis=0)[0::10], marker='.', mfc='red', mec='green', c='green', mew=2.5, ms=20, ls='', label="Method 1: Monte Carlo Sample")

            axes[i,j].hlines(y=0, xmin=0, xmax=y_test.max(), colors='k', linestyles='solid', linewidth=6, label='')#alpha=1.0)

            # plot box plot
            axes[i,j].set_ylabel("Predicted", fontsize=16)

            axes[i,j].set_xlabel(r"$\tau_{True}$", fontsize=16)

            # set limits on axes
            #axes[i,j].set_xlim(0.0475,0.0675)
            #axes[i,j].set_xlim(0.0475,0.0675)
            #axes[i,j].set_ylim(0.0475,0.0675)
            #axes[i,j].set_ylim(0.0475,0.0675)

            # set scale on axes
            axes[i,j].set_xscale('linear')
            axes[i,j].set_xscale('linear')
            axes[i,j].set_yscale('linear')
            axes[i,j].set_yscale('linear')
            
            plt.setp(axes[i,j].get_yticklabels(), fontsize=16)
            plt.setp(axes[i,j].get_xticklabels(), fontsize=16)


            #axes[0,0].legend(markerscale=2.5)
            idx +=1

    plt.legend(markerscale=1.5, fontsize=16)
    plt.tight_layout()
    
    # save plots as png and pdf
    plt.savefig(os.path.join(path, filter,
        "plot_deterministic_model_error2_{}_{}.png".format(trainig_data_name, model_name)))
    plt.savefig(os.path.join(path, filter,
        "plot_deterministic_model_error2_{}_{}.pdf".format(trainig_data_name, model_name)))
        
    return


def plot_error2(trainig_data_name, model_name, yhat, mc_predictions, y_test, path, filter="nowedge"):

    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    Y_TEST : ndarray
        Image labels. Will be converted to int. Length 1.
    YHAT : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from models trained with a DistributionLambda layer at the end.
    MC_PREDICTIONS : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from MC-Sampled models. I did NOT average the predicted values.
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    eg.
        plot_error2(trainig_data_name="toymodel",
        yhat=np.array([model(X_test),model(X_test),model(X_test),model(X_test)]),
         mc_predictions=np.array([pred_distLambda1,pred_distLambda2,pred_distLambda3,pred_distLambda4]), y_test=y_test, path="./", filter="nowedge")

    """
    
    fig, axes = plt.subplots(figsize=(15,10),nrows=2, ncols=2, sharex=False, sharey=False)
    # plt.title("Monte Carlo vs Sigma in output layer", size = 16)
    idx = 0
    for i in range(0,2): # image row
        for j in range(0,2): # image column
            # idx selects the architectures
            
            #
            axes[i,j].yaxis.tick_left()
            axes[i,j].yaxis.set_label_position("right")

            axes[i,j].tick_params(which='both', width=2)
            axes[i,j].tick_params(which='major', length=7)
            axes[i,j].tick_params(which='minor', length=4, color='r')

            """
            # plot error
            axes[i,j].errorbar(y_test[0::10], (np.mean(mc_predictions[idx], axis=0)-y_test)[0::10], yerr=np.std(mc_predictions[idx], axis=0)[0::10], marker='.', mfc='red', mec='green', c='green', mew=2.5, ms=20, ls='', label="Method 1: Monte Carlo Sample")

            axes[i,j].errorbar(y_test[0::10], yhat[idx].mean().numpy()[0::10,0]-y_test[0::10], yerr=yhat[idx].stddev().numpy()[0::10,0], marker='.', mfc='pink', mec='blue', c='blue', mew=2.5, ms=20, ls='', label="Method 2: DistributionLambda")
            """
            # plot error
            axes[i,j].errorbar(y_test[:], (np.mean(mc_predictions[idx], axis=0)-y_test)[:], yerr=np.std(mc_predictions[idx], axis=0)[:], marker='.', mfc='red', mec='green', c='green', mew=2.5, ms=20, ls='', label="Method 1: Monte Carlo Sample")

            axes[i,j].errorbar(y_test[:], yhat[idx].mean().numpy()[:,0]-y_test[:], yerr=yhat[idx].stddev().numpy()[:,0], marker='.', mfc='pink', mec='blue', c='blue', mew=2.5, ms=20, ls='', label="Method 2: DistributionLambda")

            axes[i,j].hlines(y=0, xmin=0, xmax=y_test.max(), colors='k', linestyles='solid', linewidth=6, label='')#alpha=1.0)

            # plot box plot
            axes[i,j].set_ylabel("Predicted", fontsize=16)

            axes[i,j].set_xlabel(r"$\tau_{True}$", fontsize=16)

            # set limits on axes
            #axes[i,j].set_xlim(0.0475,0.0675)
            #axes[i,j].set_xlim(0.0475,0.0675)
            #axes[i,j].set_ylim(0.0475,0.0675)
            #axes[i,j].set_ylim(0.0475,0.0675)

            # set scale on axes
            axes[i,j].set_xscale('linear')
            axes[i,j].set_xscale('linear')
            axes[i,j].set_yscale('linear')
            axes[i,j].set_yscale('linear')
            
            plt.setp(axes[i,j].get_yticklabels(), fontsize=16)
            plt.setp(axes[i,j].get_xticklabels(), fontsize=16)

            #axes[0,0].legend(markerscale=2.5)
            idx +=1

    plt.legend(markerscale=1.5, fontsize='large')
    plt.tight_layout()
    
    # save plots as png and pdf
    plt.savefig(os.path.join(path, filter,
        "plot_error2_{}_{}.png".format(trainig_data_name, model_name)))
    plt.savefig(os.path.join(path, filter,
        "plot_error2_{}_{}.pdf".format(trainig_data_name, model_name)))
        
    return
        


def plot_distribution_mean_std(trainig_data_name, model_name, yhat, mc_predictions, y_test, path, filter="nowedge"):

    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    Y_TEST : ndarray
        Image labels. Will be converted to int. Length 1.
    YHAT : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from models trained with a DistributionLambda layer at the end.
    MC_PREDICTIONS : array
        4 different model architectures but any number of predictions because the tuning params per architecture can be any number. Length of 4. Each prediction is from MC-Sampled models. I did NOT average the predicted values.
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    eg.
        plot_distribution_mean_std(trainig_data_name="toymodel",
        yhat=np.array([model(X_test),model(X_test),model(X_test),model(X_test)]),
         mc_predictions=np.array([pred_distLambda,pred_distLambda,pred_distLambda,pred_distLambda]), path="./", filter="nowedge")


    """
    for p in range(mc_predictions.shape[0]): # image row
        fig, axes = plt.subplots(figsize=(15,10),nrows=1, ncols=1, sharex=False, sharey=False)
        
        sns.distplot(yhat[p].mean(),label="DistributionLambda")
        sns.distplot(np.mean(mc_predictions[p], axis=0), label= "MC")
        plt.title("DenseFlipout mean",size = 16 )


        plt.figure(figsize=(15, 10))
        sns.distplot(yhat[p].stddev(),label="DistributionLambda")
        sns.distplot(np.std(mc_predictions[p], axis=0), label= "MC")
        plt.title("DenseFlipout stddev",size = 16 )
        
        plt.setp(axes.get_yticklabels(), fontsize=16)
        plt.setp(axes.get_xticklabels(), fontsize=16)
    
        plt.legend(markerscale=2.5, fontsize='large')
        plt.tight_layout()
        
        # save plots as png and pdf
        plt.savefig(os.path.join(path, filter,
            "plot_distribution_mean_std_{}_{}.png".format(trainig_data_name, str(p+1))))
        plt.savefig(os.path.join(path, filter,
            "plot_distribution_mean_std_{}_{}.png".format(trainig_data_name, str(p+1))))
            
    return
