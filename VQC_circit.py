#VQC_circit

import os
import torch
import pandas as pd
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import random
from torchinfo import summary
from pennylane.qnn import TorchLayer

# For reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ===== Parameters =====
depth = 8           # Depth of the circuit
n_input = 1         # Number of input points (window size)
total_points = 500  # Total number of simulated Mackey Glass points
train_ratio = 0.7
num_epochs = 300
learning_rate = 0.01
delta_huber = 0.1         # Parameter for the Huber loss
accuracy_threshold = 0.1  # Threshold to define a "correct" prediction

# Creating the directory to save results
results_dir = "MG_VQC_Results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ===== Generation of the Mackey Glass series =====
def generate_mackey_glass(total_steps, tau=17, n=10, beta=0.2, gamma=0.1, dt=1):
    """
    Simulation of the Mackey Glass series using the Euler method.
    The initial conditions (for t < tau) are set to 1.2.
    """
    history = np.zeros(total_steps + tau)
    history[:tau] = 1.2
    for t in range(tau, total_steps + tau):
        history[t] = history[t-1] + dt * (beta * history[t-tau] / (1 + history[t-tau]**n) - gamma * history[t-1])
    return history[tau:]

mackey_series = generate_mackey_glass(total_points)

# ===== Normalization of the series between -1 and 1 =====
min_val = np.min(mackey_series)
max_val = np.max(mackey_series)
mackey_series_norm = 2 * (mackey_series - min_val) / (max_val - min_val) - 1

# ===== Creation of sliding windows =====
def create_windows(series, window_size):
    X = []
    Y = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        Y.append(series[i+window_size])
    return np.array(X), np.array(Y)

X, Y = create_windows(mackey_series_norm, n_input)

# Conversion to torch.tensor (dtype=torch.float64 for consistency with PennyLane's device)
X = torch.tensor(X, dtype=torch.float64)
Y = torch.tensor(Y, dtype=torch.float64)

# Splitting into training and testing sets
split_idx = int(len(X) * train_ratio)
X_train, Y_train = X[:split_idx], Y[:split_idx]
X_test, Y_test = X[split_idx:], Y[split_idx:]

# ===== VQC with angle encoding and circular entanglement =====
def VQC(x, params):
    # Angle encoding
    qml.templates.AngleEmbedding(x, wires=range(n_input), rotation="Y")
    
    # Variational layers with RX and RY rotations + entanglement
    for i in range(depth):
        # Apply RX and RY rotations separately to use 2 parameters per qubit
        for j in range(n_input):
            qml.RX(params[i][2 * j], wires=j)
            #qml.RY(params[i][2 * j + 1], wires=j)
        
        # Circular entanglement layer (WITHOUT extra parameters)
        entangler_params = params[i][n_input:].reshape(1, n_input)
        qml.templates.BasicEntanglerLayers(entangler_params, wires=range(n_input))
    
    return [qml.expval(qml.PauliZ(j)) for j in range(n_input)]


class HybridVQC(nn.Module):
    def __init__(self):
        super().__init__()        
        self.fc = nn.Linear(n_input, 1)  # FC layer: Maps n_input → 1 output
        self.fc.weight = nn.Parameter(self.fc.weight.double())  # Convert weights to float64
        self.fc.bias = nn.Parameter(self.fc.bias.double())      # Convert bias to float64
        
        self.dev = qml.device("default.qubit", wires=n_input)
        weight_shapes = {"weights": (depth, 2 * n_input)}
        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            return VQC(inputs, weights)
        
        # Wrap the QNode in a TorchLayer.
        self.qlayer = TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        q_output = self.qlayer(x)  # Already a torch tensor with autograd support        
        q_output = q_output.view(1, -1)  # Reshape for FC layer (batch_size, n_input)
        return self.fc(q_output)  # Fully connected layer maps to a single output

# ===== Model Summary and Circuit Drawing =====
def print_model_summary(model, sample_input, dname):
    """
    Print the quantum circuit summary and save the visualization.
    """
    qnode = model
    draw_func = qml.draw_mpl(qnode, decimals=2, style="pennylane")
    
    # Generate a dummy parameter set
    dummy_weights = torch.rand(depth, 2 * n_input, dtype=torch.float64)
    
    # Draw the circuit
    fig, ax = draw_func(sample_input)
    fig.suptitle("Quantum Circuit Diagram")
    
    # Save the diagram
    save_path = os.path.join(results_dir,"circuit_visuals", f"VQC_circuit_{dname}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved circuit diagram at {save_path}")

#model
hybrid_vqc_model = HybridVQC()

num_params = sum(p.numel() for p in hybrid_vqc_model.parameters() if p.requires_grad)
print(f"VQC - Number of trainable parameters: {num_params}")


# Optimizer
optimizer_vqc = torch.optim.Adam(hybrid_vqc_model.parameters(), lr=learning_rate)

# ===== Loss function and metrics =====
def huber_loss(y_true, y_pred, delta=delta_huber):
    error = y_true - y_pred
    is_small_error = torch.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(is_small_error, squared_loss, linear_loss)

def loss_fn(model, params, X, Y, delta=delta_huber):  
    preds = torch.stack([model(x) for x in X])  # Stack outputs  
    return torch.mean(huber_loss(Y, preds.squeeze(), delta))  # Squeeze to match target shape

def accuracy_fn(model, params, X, Y, threshold=accuracy_threshold):
    preds = torch.stack([model(x) for x in X])
    correct = torch.sum(torch.abs(preds.squeeze() - Y) < threshold)
    return correct.item() / len(Y)


# Draw the circuit visualization
sample_input = torch.rand(n_input, dtype=torch.float64)
print_model_summary(hybrid_vqc_model, sample_input, "Mackey_Glass")
m_summary = summary(hybrid_vqc_model, input_size=(1, n_input), dtypes=[torch.float64])
#m_summary = 0
print(m_summary)
#m_summary = summary(hybrid_vqc_model(X_train[0]), input_size=sample_input.shape)

# ===== Training loop =====
train_loss_vqc, test_loss_vqc = [], []
train_acc_vqc, test_acc_vqc = [], []
train_preds_vqc, test_preds_vqc = [], []
grad_norm_vqc = []

log_file_path = os.path.join(results_dir, "metrics_vqc.txt")
log_file = open(log_file_path, "w", encoding = 'utf-8')
log_file.write("Mackey Glass Time Series Prediction using VQC\n")
log_file.write(f"Depth: {depth}, Input Points: {n_input}, Total Points: {total_points}\n")
log_file.write(f"Train Ratio: {train_ratio}, Epochs: {num_epochs}, Learning Rate: {learning_rate}\n")
log_file.write(f"Delta Huber: {delta_huber}, Accuracy Threshold: {accuracy_threshold}\n\n")
log_file.write("VQC Parameters: " + str(num_params) + "\n\n")
log_file.write("Model Summary: " + str(m_summary) + "\n\n")
log_file.write("epoch, train_loss_vqc, test_loss_vqc, train_acc_vqc, test_acc_vqc, grad_norm_vqc\n")
 

log_file.flush()

for epoch in range(num_epochs):
    optimizer_vqc.zero_grad()
    loss_vqc = loss_fn(hybrid_vqc_model, hybrid_vqc_model.parameters(), X_train, Y_train)
    loss_vqc.backward()
    total_norm_vqc = 0.0
    for p in hybrid_vqc_model.parameters():
        if p.grad is not None:
            total_norm_vqc += p.grad.data.norm(2).item() ** 2
    grad_norm_val_vqc = total_norm_vqc ** 0.5
    
    optimizer_vqc.step()
    
    # Evaluate metrics
    current_train_loss_vqc = loss_fn(hybrid_vqc_model, hybrid_vqc_model.parameters(), X_train, Y_train).item()
    current_test_loss_vqc = loss_fn(hybrid_vqc_model, hybrid_vqc_model.parameters(), X_test, Y_test).item()
    current_train_acc_vqc = accuracy_fn(hybrid_vqc_model, hybrid_vqc_model.parameters(), X_train, Y_train)
    current_test_acc_vqc = accuracy_fn(hybrid_vqc_model, hybrid_vqc_model.parameters(), X_test, Y_test)

    # Predictions for visualization    
    preds_train_vqc = torch.stack([hybrid_vqc_model(x) for x in X_train]).detach().numpy()
    preds_test_vqc = torch.stack([hybrid_vqc_model(x) for x in X_test]).detach().numpy()
    
    train_loss_vqc.append(current_train_loss_vqc)
    test_loss_vqc.append(current_test_loss_vqc)
    train_acc_vqc.append(current_train_acc_vqc)
    test_acc_vqc.append(current_test_acc_vqc)
    grad_norm_vqc.append(grad_norm_val_vqc)
    train_preds_vqc.append(preds_train_vqc)
    test_preds_vqc.append(preds_test_vqc)    
    df_train_pred_vqc = pd.DataFrame({
    "Y_train": Y_train.detach().numpy().flatten(),
    "train_preds_vqc": train_preds_vqc[-1].squeeze(),    
    })
    df_test_pred_vqc = pd.DataFrame({    
    "Y_test": Y_test.detach().numpy().flatten(),
    "test_preds_vqc": test_preds_vqc[-1].squeeze(),
    })
    
    log_line = (f"{epoch}, {current_train_loss_vqc:.6f}, {current_test_loss_vqc:.6f}, "
                f"{current_train_acc_vqc:.6f}, {current_test_acc_vqc:.6f}, {grad_norm_val_vqc:.6f}\n")
    log_file.write(log_line)
    log_file.flush()
    
    print(f"Epoch {epoch}: Loss: {current_train_loss_vqc:.4f}, Test Loss: {current_test_loss_vqc:.4f}, "
          f"Train Acc: {current_train_acc_vqc:.4f}, Test Acc: {current_test_acc_vqc:.4f}, Grad Norm: {grad_norm_val_vqc:.4f}")

log_file.close()

df_train_pred_vqc.to_csv("MG_VQC_Results/VQC_train_predictions.csv", index=False)
df_test_pred_vqc.to_csv("MG_VQC_Results/VQC_test_predictions.csv", index=False)

# ===== Plotting Results =====
epochs = range(num_epochs)
plt.figure()
plt.plot(epochs, train_loss_vqc, label="Train Loss VQC")
plt.plot(epochs, test_loss_vqc, label="Test Loss VQC")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs (Mackey Glass, VQC)")
plt.legend()
plt.savefig(os.path.join(results_dir, "loss_vqc.png"), dpi=300)
plt.close()

plt.figure()
plt.plot(epochs, train_acc_vqc, label="Train Acc VQC")
plt.plot(epochs, test_acc_vqc, label="Test Acc VQC")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs (Mackey Glass, VQC)")
plt.legend()
plt.savefig(os.path.join(results_dir, "accuracy_vqc.png"), dpi=300)
plt.close()

plt.figure()
plt.plot(epochs, grad_norm_vqc, label="Grad Norm VQC")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm over Epochs (Mackey Glass, VQC)")
plt.legend()
plt.savefig(os.path.join(results_dir, "grad_norm_vqc.png"), dpi=300)
plt.close()

model_path = os.path.join(results_dir, "VQC_preTrained.pt")
torch.save(hybrid_vqc_model.state_dict(), model_path)
print("Training complete. Metrics and figures saved in", results_dir)