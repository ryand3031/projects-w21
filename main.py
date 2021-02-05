import argparse
import os
import time


import constants
from datasets.StartingDataset import StartingDataset
from networks.ResNextModel import ResNextModel
from train_functions.starting_train import starting_train
import torch


SUMMARIES_PATH = "training_summaries"


def main():
    # Get command line arguments
    args = parse_arguments()
    hyperparameters = {"epochs": args.epochs, "batch_size": args.batch_size}

    # Create path for training summaries
    label = f"cassava__{int(time.time())}"
    summary_path = f"{SUMMARIES_PATH}/{label}"
    os.makedirs(summary_path, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Summary path:", summary_path)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)

    # Initalize dataset and model. Then train the model!
    train_dataset = StartingDataset(train=True)
    val_dataset = StartingDataset(train=False)
    model = ResNextModel(5).to(device)
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=args.n_eval,
        summary_path=summary_path,
        device=device,
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=constants.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument(
        "--n_eval", type=int, default=constants.N_EVAL,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
