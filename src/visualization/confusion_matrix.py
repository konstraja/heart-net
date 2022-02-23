# -*- coding: utf-8 -*-

"""
Computes the confusion matrix for a trained model. Takes as input important parameters such as the benchmark studied.
A confusion matrix in .png format is saved in the trained model folder provided.
Plotting the confusion matrix is best called using the provided Makefile provided.
>>> make confusion_matrix_cpkl \\
    MODEL_FOLDER=X \\
    CPKL_FILE=X \\
With the parameters taking from the following values :
    - MODEL_FOLDER:
        Output path to save models and log files. A folder inside that path will be automatically created. Default
        location is *./models/*
    - CPKL_FILE:
        Pickle file containing ground truthes and predictions.
"""
import pickle
from sklearn.metrics import cohen_kappa_score
import logging
import click
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import matplotlib.pyplot as plt


from src.models.training_utils import evaluate_per_action_type, plot_confusion_matrix

@click.command()
@click.argument('model_folder',  type=click.Path(exists=True))
@click.argument('pickel_file', type=click.Path())
def main(model_folder, pickel_file):

    # Print summary
    print("\r\n\n\n========== PLOT CONFUSION MATRIX ==========")
    print("-> model_folder : " + model_folder)
    print("-> cpkl_file : " + str(pickel_file))

    pickle_test = open(model_folder + pickel_file, 'rb')
    data = pickle.load(pickle_test)
    pickle_test.close()
    y_true, y_pred = data

    evaluate_per_action_type(y_true, y_pred)

    plot_confusion_matrix(y_true, y_pred, ["A43", "A45"], normalize=True, title="Confusion matrix")
    plt.savefig(model_folder  + "cm.png")

    print("Cohen's kappa : " + str(cohen_kappa_score(y_true, y_pred)))

    # echo -en "\e[?25h"
    print("-> Done !")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()