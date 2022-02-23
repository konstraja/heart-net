# -*- coding: utf-8 -*-
from datetime import datetime
import logging
import os
import random
from typing import Callable
import click
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import torch

from src.data.data_loaders import get_data_loaders
from src.models.model import FUSION
from src.models.training_utils import train_model, device

@click.command()
@click.argument('data_path',  type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--epochs', '-e', type=int, default=10)
@click.option('--batch_size', '-b', type=int, default=32)
@click.option('--seed', '-s', type=int, default=0)
@click.option('--samples', type=click.Path(), default="sample_names.txt")
@click.option('--pretrained', type=bool, default=True)
@click.option('--optimiser', "-o", type=str, default="ADAM")
@click.option('--learning_rate', "-lr", type=float, default=1e-4)
def main(data_path, output_path, epochs, batch_size, seed, samples, pretrained, optimiser, learning_rate):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Received training request ")
    logger.info(f"Input data path: {data_path} Output folder: {output_path}")
    logger.info(f"Epochs: {epochs} Batch Size: {batch_size}")

    initialise(seed)

    training_generator, validation_generator, test_generator = get_data_loaders(data_path, f"{data_path}/{samples}", batch_size)
    logger.info(f"Created Loaders")

    neural_net = FUSION(pretrained)
    
    # If multiple GPUs are available (not guaranteed to work)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        neural_net = torch.nn.DataParallel(neural_net)

    # Push model to first GPU if available
    neural_net.to(device)

    output_path += "pretrained=" + str(pretrained)

    # Create folder if does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Start training
    train_model(neural_net,
                optimiser,
                learning_rate,
                0,
                0,
                epochs,
                1,
                True,
                output_path,
                training_generator,
                test_generator,
                validation_generator)

    # echo -en "\e[?25h"
    print("-> Done !")


def initialise(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()