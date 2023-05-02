from collections import defaultdict
from datetime import timedelta
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from tqdm.auto import tqdm


class cross_validation_model(nn.Module):
    def __init__(self, input_size=2, output_size=1):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 32)
        self.hidden_layer1 = nn.Linear(32, 64)
        self.hidden_layer2 = nn.Linear(64, 128)
        self.hidden_layer3 = nn.Linear(128, 64)
        self.hidden_layer4 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, output_size)
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.bn1(self.tanh(self.input_layer(x)))
        x = self.bn2(self.tanh(self.hidden_layer1(x)))
        x = self.bn3(self.tanh(self.hidden_layer2(x)))
        x = self.bn4(self.tanh(self.hidden_layer3(x)))
        x = self.bn5(self.tanh(self.hidden_layer4(x)))
        x = self.output_layer(x)
        return x


class train_all_model(nn.Module):
    def __init__(self, input_size=2, output_size=1):
        super().__init__()
        layers = []
        neurons = [32, 64, 128, 64] * 3 + [32, 16, 8, 4, 2]
        for neuron in neurons:
            layers.append(nn.Linear(input_size, neuron))
            layers.append(nn.Tanh())
            layers.append(nn.BatchNorm1d(neuron))
            input_size = neuron
        layers.append(nn.Linear(input_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Define the training step:
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    # Set the model to train mode
    model.train()
    # Set the training loss to 0
    train_loss = 0
    # Iterate over the DataLoader batches
    for batch, (X, y) in enumerate(dataloader):
        # Move the batch to the device
        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_pred = model(X)
        # 2. Calculate and accumulate loss
        loss = loss_func(y_pred, y)
        train_loss += loss.item()
        # 3. Zero the gradients
        optimizer.zero_grad()
        # 4. Backward pass
        loss.backward()
        # 5. Update the parameters
        optimizer.step()
    # Return average loss
    return train_loss / len(dataloader)


# Define the validating step:
# Turn on inference context manager
@torch.inference_mode()
def valid_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_func: torch.nn.Module,
    device: torch.device,
) -> float:
    # Set the model to eval mode
    model.eval()
    # Set the validating loss to 0
    valid_loss = 0
    # Iterate over the DataLoader batches
    for batch, (X, y) in enumerate(dataloader):
        # Move the batch to the device
        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_pred = model(X)
        # 2. Calculate and accumulate loss
        valid_loss += loss_func(y_pred, y).item()
    # Return average loss
    return valid_loss / len(dataloader)


# Define the training and validating loops:
def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    epochs: int,
    device: torch.device,
) -> defaultdict:
    # Init the results
    result = defaultdict(list)
    # Set the model to the device
    model.to(device)
    # Iterate over the epochs
    for epoch in tqdm(range(1, epochs + 1)):
        # Train the model
        train_loss = train_step(model, train_dataloader, loss_func, optimizer, device)
        # Validate the model
        valid_loss = valid_step(model, valid_dataloader, loss_func, device)
        # Record the loss
        result["train_loss"].append(train_loss)
        result["valid_loss"].append(valid_loss)
        # Adjust the learning rate
        if scheduler:
            scheduler.step(valid_loss)
    # Return the results
    return result


def plot(result):
    train_loss = result["train_loss"]
    valid_loss = result["valid_loss"]
    epochs = range(len(result["train_loss"]))
    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, valid_loss, label="valid_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# Load the data from the CSV file
def cross_validation_train(path="./train.csv"):
    # Set the device, epochs, batch size and learning rate
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 500
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.002
    # Init the models list to save the models
    models = []
    # Read the training data
    df = pd.read_csv(path)
    # Get the X and y values
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32).reshape(-1, 2)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    # Create the dataset
    dataset = TensorDataset(X, y)
    # Split the data into 5 folds
    kfold = KFold(n_splits=5)
    start_time = timer()
    # For each fold, train a model and save the model
    for i, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        # Split into training and validation sets
        train_subsampler = SubsetRandomSampler(train_idx)
        valid_subsampler = SubsetRandomSampler(test_idx)
        train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
        valid_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_subsampler)
        # Create the model, loss function, optimizer, and scheduler
        model = cross_validation_model()
        loss_func = nn.HuberLoss(reduction="mean")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.5)
        scheduler = scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)
        # Train the model for EPOCHS
        result = train(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=EPOCHS,
            device=DEVICE,
        )
        print(f"KFold: {i+1} Train loss: {result['train_loss'][-1]} Test loss: {result['valid_loss'][-1]}")
        models.append(model.state_dict())
        # plot(result)
    end_time = timer()
    print(f"Total training time: {timedelta(seconds=end_time-start_time)}")
    return models


def train_all(path="./train.csv"):
    # Set device, epochs, and learning rate
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 20000
    LEARNING_RATE = 0.002
    # Read the training data
    df = pd.read_csv(path)
    # Get the X and y values
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    # Convert to tensors
    X = torch.tensor(X, device=DEVICE, dtype=torch.float32).reshape(-1, 2)
    y = torch.tensor(y, device=DEVICE, dtype=torch.float32).reshape(-1, 1)
    # Create the model, loss function, optimizer, and scheduler
    model = train_all_model().to(DEVICE)
    loss_func = nn.HuberLoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.6)
    scheduler = scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)
    train_loss = []
    # Train the model for EPOCHS
    start_time = timer()
    for epoch in tqdm(range(EPOCHS)):
        y_pred = model(X)
        loss = loss_func(y_pred, y)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} Loss: {loss.item()}")
        scheduler.step(loss)
    end_time = timer()
    print(f"Total training time: {timedelta(seconds=end_time-start_time)}")
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss)
    plt.title("Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    return model


# Turn on inference context manager
@torch.inference_mode()
def cross_validation_predict(models, path="./test.csv"):
    # Set device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Read test data
    X = pd.read_csv(path).iloc[:, 1:].values
    # Convert test data to tensor
    X = torch.tensor(X, dtype=torch.float32).reshape(-1, 2).to(DEVICE)
    # Initialize model and set to device
    model = cross_validation_model().to(DEVICE)
    # Make predictions
    preds = np.zeros(len(X))
    for state_dict in models:
        # Load model state dict and set to eval mode
        model.load_state_dict(state_dict)
        model.eval()
        # Convert tensor to numpy array
        preds += model(X).cpu().detach().numpy().reshape(-1)
    # Average predictions
    preds /= len(models)
    # Create submission file
    submission = pd.DataFrame({"id": range(1, len(preds) + 1), "y": preds})
    # Save submission file
    submission.to_csv("./submission.csv", index=False)


# Turn on inference context manager
@torch.inference_mode()
def train_all_predict(model, path="./test.csv"):
    # Set device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set model to device
    model.to(DEVICE)
    # Set model to eval mode
    model.eval()
    # Read test data
    X = pd.read_csv(path).iloc[:, 1:].values
    # Convert test data to tensor
    X = torch.tensor(X, dtype=torch.float32).reshape(-1, 2).to(DEVICE)
    # Predict test data
    preds = model(X)
    # Convert tensor to numpy array
    preds = preds.detach().cpu().numpy().reshape(-1)
    # Create submission file
    submission = pd.DataFrame({"id": range(1, len(preds) + 1), "y": preds})
    # Save submission file
    submission.to_csv("./submission.csv", index=False)


def calculate_mse(y_pred_path="./submission.csv", y_best_path="./sample.csv"):
    y_pred = pd.read_csv(y_pred_path)
    y_best = pd.read_csv(y_best_path)
    y_pred = y_pred.iloc[:, -1].values
    y_best = y_best.iloc[:, -1].values
    return np.mean((y_pred - y_best) ** 2)


if __name__ == "__main__":
    TRAIN_CSV_PATH = "./train.csv"
    SAMPLE_CSV_PATH = "./sample.csv"
    TEST_CSV_PATH = "./test.csv"
    # models = cross_validation_train(TRAIN_CSV_PATH)
    # cross_validation_predict(models, TEST_CSV_PATH)
    # model = train_all()
    # train_all_predict(model, TEST_CSV_PATH)
    # print(calculate_mse(SAMPLE_CSV_PATH, "./submission.csv"))
