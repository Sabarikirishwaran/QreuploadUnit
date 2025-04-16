# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:46:51 2025

@author: casse
"""

import os
import random
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pennylane.qnn import TorchLayer

# Configuration 
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Parameters
nb_epoch = 300
lr = 0.01
seq_length = 3
early_stop_patience = 20  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
# Data Generation
#############################################

def preprocess_river_data(file_path):
    
    df = pd.read_csv(file_path, skiprows=2)
    df.columns = ['date', 'wlvalue', 'fvalue']
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=False)
    df = df.dropna(subset=['date'])
    df.set_index('date', inplace=True)
    
    df['wlvalue_normalized'] = 2 * (df['wlvalue'] - df['wlvalue'].min()) / (df['wlvalue'].max() - df['wlvalue'].min()) - 1
    return df[['wlvalue_normalized']]

def generate_mackey_glass_data(num_points=1000, delta_t=1.0, tau=17, beta=0.2, gamma=0.1, n=10):
    
    history = 1.2
    x = [history] * (tau + 1)
    for t in range(num_points):
        x_new = beta * x[-tau-1] / (1 + x[-tau-1]**n) - gamma * x[-1]
        x.append(x[-1] + x_new * delta_t)
    x = np.array(x[tau+1:])
    # Normalisation dans [-1, 1]
    x_normalized = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
    df = pd.DataFrame({'wlvalue_normalized': x_normalized})
    df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    return df


def generate_sin_data(num_points=1000):
    """
    Generate a sine wave of a given length, amplitude, period, and optional noise.
    Returns a 1D numpy array of shape (length,).
    """
    length = num_points
    frequency = 10
    amplitude = 1
    sampling_rate = 1.0

    # Generate time values
    t = np.linspace(0, length / sampling_rate, length)

    # Generate sine wave
    x = amplitude * np.sin(2 * np.pi * frequency * t)  
    
    x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
    df = pd.DataFrame({'wlvalue_normalized': x_normalized})
    df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D') 

    return df

#############################################
# Series generation
#############################################

def create_sequences(data, seq_length=3):

    X, y = [], []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length, 0].values
        X.append(seq)
        y.append(data.iloc[i + seq_length, 0])
    return np.array(X), np.array(y)

def split_data(data, test_size=0.2, seq_length=3):

    X, y = create_sequences(data, seq_length=seq_length)
    train_size = int((1 - test_size) * len(X))
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

#############################################
# QuantumNet Model
#############################################

class QuantumNet(nn.Module):
    def __init__(self, seq_length, circuit_depth, n_qubits, alpha=0.5, output_dim=1):        
        super(QuantumNet, self).__init__()
        self.seq_length = seq_length
        self.circuit_depth = circuit_depth
        self.alpha = alpha
        self.n_qubits = n_qubits  # Data qubit + ancilla qubits
        
        # output linear layer
        self.fc_out = nn.Linear(1, output_dim)
        
        weight_shapes = {"weights": (1, n_qubits, self.circuit_depth, 5*3)}                
                
        self.num_wires = self.n_qubits + (self.n_qubits * circuit_depth)
        self.dev = qml.device("default.qubit", wires=self.num_wires)
        
        def quantum_circuit(inputs, weights):            
                
            for k in range(self.n_qubits):                
                # --- For each encoding block, perform data operations controlled by the corresponding ancilla ---
                for i in range(self.circuit_depth):
                    ancilla_wire = self.n_qubits + (k * self.circuit_depth) + i
                    qml.Rot(weights[0][k][i][0], weights[0][k][i][1], weights[0][k][i][2], wires=ancilla_wire)

                    # Pre-encoding rotation on the data qubit.
                    qml.Rot(weights[0][k][i][3], weights[0][k][i][4], weights[0][k][i][5], wires=k)
                    
                    # Data encoding: apply RY rotations for each input element.                    
                    for j in range(seq_length):
                        qml.RY(inputs[j], wires=k)                    
                    
                    qml.CRot(weights[0][k][i][6], weights[0][k][i][7]*inputs[-1], weights[0][k][i][8], wires=[ancilla_wire, k])                                        
                    
                    # Post-encoding rotation on the data qubit.                    
                    qml.Rot(weights[0][k][i][9], weights[0][k][i][10], weights[0][k][i][11], wires=k)
                    
                    qml.Rot(weights[0][k][i][-3], weights[0][k][i][-2], weights[0][k][i][-1], wires=ancilla_wire)
                                   
            proj = (qml.Identity(wires=self.n_qubits) + qml.PauliZ(wires=self.n_qubits)) / 2

            for m in range(self.n_qubits+1, self.num_wires):
                proj = proj @ (qml.Identity(wires=m) + qml.PauliZ(wires=m)) / 2             

            dmeasure = qml.PauliZ(0)
            for n in range(1, self.n_qubits):
                dmeasure = dmeasure @ qml.PauliZ(n)

            return qml.expval(dmeasure @ proj)
        
        # Wrap the quantum circuit in a QNode and then in a TorchLayer.
        @qml.qnode(self.dev, interface="torch")
        def qnode_circuit(inputs, weights):
            return quantum_circuit(inputs, weights)
        
        self.qlayer = TorchLayer(qnode_circuit, weight_shapes)
        
    def forward(self, x):        
        outputs = []
        for sample in x:
            outputs.append(self.qlayer(sample))
        q_out = torch.stack(outputs)
        #q_out = q_out.reshape(-1, seq_length)        
        q_out = q_out.unsqueeze(1)
        out = self.fc_out(q_out)        
        return out.squeeze()


#############################################
# Metrics
#############################################

def huber_loss(y_pred, y_true, delta=1.0):
    diff = torch.abs(y_pred - y_true)
    loss = torch.where(diff <= delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    return torch.mean(loss)

def prediction_accuracy(y_pred, y_true, tolerance=0.1):
    
    within_tolerance = torch.abs(y_pred - y_true) <= (tolerance * torch.abs(y_true))
    return torch.mean(within_tolerance.float()) * 100

#############################################
# Training
#############################################

def train_model(X_train, y_train, model, results_dir):     
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    metrics_file = os.path.join(results_dir, "metrics.csv")
    criterion = nn.MSELoss()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    print(f"Tot.param_count: {param_count}")

    with open(metrics_file, "w") as f:
        f.write("epoch,loss,accuracy,grad_norm\n")
        
    best_loss = float('inf')
    patience_counter = 0    

    # Sauvegarde initiale du modèle
    best_model_state = model.state_dict()

    for epoch in range(nb_epoch):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        #loss = huber_loss(y_pred, y_train)        
        loss = criterion(y_pred, y_train)
        
        loss.backward()
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5
        
        optimizer.step()
        
        accuracy = prediction_accuracy(y_pred, y_train)
        
        with open(metrics_file, "a") as f:
            f.write(f"{epoch + 1},{loss.item()},{accuracy.item()},{grad_norm}\n")            
        print(f"Epoch {epoch + 1}/{nb_epoch} - Loss: {loss.item():.4f} - Accuracy: {accuracy.item():.2f}%")
                

        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at {epoch + 1}")
                model.load_state_dict(best_model_state)
                break

def save_predictions_progressive(X, y, model, predictions_file):
    with open(predictions_file, "w") as f:
        f.write("true_values,predictions\n")
    with open(predictions_file, "a") as f:
        for i in range(len(X)):
            prediction = model(X[i].unsqueeze(0)).detach().item()
            f.write(f"{y[i].item()},{prediction}\n")

def visualize_metrics(metrics_file, output_file):
    metrics_df = pd.read_csv(metrics_file)
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    plt.plot(metrics_df["epoch"], metrics_df["loss"], label="Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(metrics_df["epoch"], metrics_df["accuracy"], label="Accuracy (%)")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Gradient norm subplot
    plt.subplot(1, 3, 3)
    plt.plot(metrics_df["epoch"], metrics_df["grad_norm"], label="Gradient Norm", color="red")
    plt.title("Gradient Norm per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def visualize_predictions(predictions_file, y_true, output_file, title):
    predictions_df = pd.read_csv(predictions_file)
    true_values = y_true.cpu().numpy()

    plt.figure(figsize=(14, 6))
    plt.plot(range(len(true_values)), true_values, label="Actual Values", color="black", linewidth=2)
    plt.plot(range(len(predictions_df["predictions"])), predictions_df["predictions"], label="predictions")
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.savefig(output_file, dpi=300)
    plt.close()

def print_model_summary(model, sample_input,dname):
    
    qnode = model.qlayer.qnode
    draw_func = qml.draw_mpl(qnode, decimals=2, style="pennylane")
        
    dummy_weights = torch.rand(1, model.n_qubits, model.circuit_depth, 5*3, dtype=torch.float64)
            
    fig, ax = draw_func(sample_input, dummy_weights)
    fig.suptitle("Quantum Circuit Diagram")
    
    save_path = os.path.join("circuit_visuals", f"quantum_circuit_{dname}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    print(f"Saved circuit diagram at {save_path}")

#############################################
# Experiments
#############################################

def run_experiment(dataset_name, data, circuit_depth, n_qubits):
    
    print(f"\n--- Starting of Experiment {dataset_name} ---")
    results_dir = f"QRB_implementation_{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    X_train, y_train, X_test, y_test = split_data(data, test_size=0.2, seq_length=seq_length)
    X_train = torch.tensor(X_train, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)
    X_test = torch.tensor(X_test, dtype=torch.float64)
    y_test = torch.tensor(y_test, dtype=torch.float64)
    X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)
        
    model = QuantumNet(seq_length=seq_length, circuit_depth=circuit_depth, n_qubits=n_qubits, alpha=0.5, output_dim=1).to(torch.double)
    model = model.to(device)
    
    print("Training the model :")
    print_model_summary(model, X_train[0], dataset_name)
    
    train_model(X_train, y_train, model, results_dir)
    
    train_predictions_file = os.path.join(results_dir, f"{dataset_name}_train_predictions.csv")
    test_predictions_file = os.path.join(results_dir, f"{dataset_name}_test_predictions.csv")
    save_predictions_progressive(X_train, y_train, model, train_predictions_file)
    save_predictions_progressive(X_test, y_test, model, test_predictions_file)
    
    metrics_file = os.path.join(results_dir, "metrics.csv")
    metrics_output = os.path.join(results_dir, f"{dataset_name}_loss_accuracy_comparison.png")
    visualize_metrics(metrics_file, metrics_output)
    
    train_output = os.path.join(results_dir, f"{dataset_name}_predictions_train_comparison.png")
    test_output = os.path.join(results_dir, f"{dataset_name}_predictions_test_comparison.png")
    visualize_predictions(train_predictions_file, y_train, train_output, f"Predictions vs Actual Values (Train) - {dataset_name}")
    visualize_predictions(test_predictions_file, y_test, test_output, f"Predictions vs Actual Values (Test) - {dataset_name}")

    model_path = os.path.join(results_dir, f"{dataset_name}_best_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    print(f"--- Experiment Completed {dataset_name} ---\n")

#############################################
# Exp Execution
#############################################

if __name__ == "__main__":
    
    river_file_path = "QLSTM-DRU-Environmental-TimeSeries\\QRU+QRBs\\river_level.csv"
    river_data = preprocess_river_data(river_file_path)    
    mackey_data = generate_mackey_glass_data(num_points=1000)    
    sin_data = generate_sin_data(num_points=1000)
        
    #run_experiment("River", river_data, circuit_depth=6, n_qubits=1)
    run_experiment("MackeyGlass", mackey_data, circuit_depth=3, n_qubits=1)
    #run_experiment("Sinusoide", sin_data, circuit_depth=3, n_qubits=1)
