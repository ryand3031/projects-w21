import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard
from train_functions.starting_train import evaluate, compute_accuracy

def saving_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path, device, save_dir
):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
        device:          Device that we're using
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Initialize summary writer (for logging)
    writer = torch.utils.tensorboard.SummaryWriter(summary_path)
    
    model.train()
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")

            # TODO: Backpropagation and gradient descent
            input_data, labels = batch
            input_data, labels = input_data.to(device), labels.to(device)
            optimizer.zero_grad()

            predictions = model(input_data)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()


            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:

                train_accuracy = compute_accuracy(predictions.argmax(axis = 1), labels)
                writer.add_scalar("train_accuracy", train_accuracy, global_step = step)
                writer.add_scalar("train_loss", loss, global_step = step)

                val_loss, val_accuracy = evaluate(val_loader, model, loss_fn, device)
                writer.add_scalar("val_loss", val_loss, global_step=step)
                writer.add_scalar("val_accuracy", val_accuracy, global_step=step)
                
                print(f"Eval:\t{step/n_eval}")
                print(f"Validation loss:\t{val_loss}")
                print(f"Validation Accuracy:\t{val_accuracy}")

            step += 1
        
        torch.save(model.state_dict(), f'{save_dir}/model-{datetime.now.strftime("%m-%d-%Y_%H-%M-%S")}.pt')
        print()

    print(f"Final Evaluation: {evaluate(val_loader, model, loss_fn, device)}")
