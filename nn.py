import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as data_utils
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import data_statistics

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    if use_cuda:
        print('Using CUDA!')
    else:
        print('WARNING: Could not find CUDA. Using CPU')
    
    return device


class NetModel(nn.Module):
    def __init__(self, features, classes):
        super().__init__()

        nodes = 128

        self.input = nn.Linear(features, nodes)
        self.linear1 = nn.Linear(nodes, nodes)
        self.dropout1 = nn.Dropout(.2)
        self.linear2 = nn.Linear(nodes, nodes)
        self.linear3 = nn.Linear(nodes, nodes)
        self.dropout3 = nn.Dropout(.2)
        self.output = nn.Linear(nodes, classes)

        self.activation = F.relu

    def forward(self, x):
        x = self.input(x)
        x = self.activation(x)

        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.activation(x)

        x = self.linear3(x)
        x = self.activation(x)
        x = self.dropout3(x)

        # don't need softmax because it is applied by using CrossEntropyLoss
        return self.output(x)


def evaluate_nn(X: np.ndarray, y: np.ndarray, test_size=0.1, class_names: list | None = None) -> None:
    device = get_device()
    features = X.shape[1]
    classes = np.max(y) + 1  # assumes labels 0 -> # classes - 1
    alpha = 1e-4
    batch_size = 128
    iterations = 50

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    train_data = data_utils.TensorDataset(X_train, y_train)
    train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    net_model = NetModel(features, classes)
    net_model.to(device)  # make sure the model is CUDA

    # Loss stuff
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net_model.parameters(), lr=alpha)

    tick = time.time()
    for epoch in range(iterations):
        running_loss_total = 0
        for i, data in enumerate(train_loader, 1):
            inputs, targets = data  # shoud already be on the GPU
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            y_pred = net_model(inputs)
            loss = loss_fn(y_pred, targets.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss_total += loss.item()
        
        # Get the average loss for this epoch
        avg_loss = running_loss_total / len(train_loader)
        if epoch % 1 == 0:
            print(f'Epoch {epoch} average loss: {avg_loss}')
        if avg_loss < 1e-8:
            print(f'Loss threashold reached. Stopping at epoch {epoch}')
            break

    tock = time.time()
    print(f'Training finished in {(tock - tick):.2f} seconds')

    # Test the the model
    net_model.eval()  # put the model into evaluation (testing) mode
    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        outputs = net_model(X_test)
        _, predicted = torch.max(outputs, 1)
        
        # Calculate statistics
        correct_count = (predicted == y_test).sum().item()
        accuracy = correct_count / len(y_test)

        if classes == 2:
            tp = ((predicted == 1) & (y_test == 1)).sum().item()
            fp = ((predicted == 1) & (y_test == 0)).sum().item()
            tn = ((predicted == 0) & (y_test == 0)).sum().item()
            fn = ((predicted == 0) & (y_test == 1)).sum().item()

            print('\nResults:')
            print('True Positives:', tp)
            print('False Positives:', fp)
            print('True Negatives:', tn)
            print('False Negatives:', fn)
            print()

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = (2 * precision * recall) / (precision + recall)
    
    # Print results
    print(f'Accuracy: {accuracy*100:.2f}%')
    if classes == 2:
        print(f'Precision: {precision*100:.2f}%')
        print(f'Recall: {recall*100:.2f}%')
        print(f'F1-score: {f1_score:.2f}')
    
    # Generate confusion matrix
    # Plot confusion matrix for binary classification or when class_names provided
    try:
        y_cpu = y_test.cpu().numpy()
        pred_cpu = predicted.cpu().numpy()
    except Exception:
        # fallback to tensors if cpu->numpy isn't available
        y_cpu = y_test.cpu()
        pred_cpu = predicted.cpu()

    if classes == 2:
        # ensure we have readable class names
        names = class_names if class_names is not None else ['benign', 'malicious']
        data_statistics.plot_confusion_matrix(y_cpu, pred_cpu, names)
    elif class_names is not None:
        data_statistics.plot_confusion_matrix(y_cpu, pred_cpu, class_names)
    else:
         data_statistics.plot_confusion_matrix(y_test.cpu(), predicted.cpu(), class_names = None)

