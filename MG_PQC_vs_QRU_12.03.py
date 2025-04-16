import os
import torch
import random
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary

# For reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ===== Parameters =====
depth = 3           # Depth of the circuit
n_input = 3         # Number of input points (window size)
total_points = 500  # Total number of simulated Mackey Glass points
train_ratio = 0.7
num_epochs = 300
learning_rate = 0.01
delta_huber = 0.1         # Parameter for the Huber loss
accuracy_threshold = 0.1  # Threshold to define a "correct" prediction

# Creating the directory to save results
results_dir = "MG_PQC_vs_QRU_FFT"
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

# ===== Definition of the quantum device =====
dev = qml.device("default.qubit", wires=1)

# ===== Definition of the quantum circuits =====
# QRU circuit with re-uploading
def QRU(params, x, alpha=0.5):
    # The re-uploading circuit with 3 rotations per input per layer
    for i in range(depth):
        for j in range(len(x)):
            qml.RY(x[j], wires=0)
            qml.RX(params[i][2 * j], wires=0)
            qml.RY(params[i][2 * j + 1], wires=0)
            # qml.RZ(params[i][3 * j + 2], wires=0)
    return qml.expval(qml.PauliZ(0))

# Circuit without re-uploading (PQC)
def PQC(params, x):
    # Data encoding in one shot
    for j in range(len(x)):
        qml.RY(x[j], wires=0)
    # Followed by variational layers (here 2 rotations per input per layer)
    for i in range(depth):
        for j in range(len(x)):
            qml.RX(params[i][2 * j], wires=0)
            qml.RY(params[i][2 * j + 1], wires=0)
            # qml.RZ(params[i][3 * j + 2], wires=0)
    return qml.expval(qml.PauliZ(0))

# Definition of qnodes for each circuit
@qml.qnode(dev, interface="torch")
def circuit_qru(params, x):
    return QRU(params, x)

@qml.qnode(dev, interface="torch")
def circuit_pqc(params, x):
    return PQC(params, x)


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
    fig, ax = draw_func(dummy_weights,sample_input)
    fig.suptitle("Quantum Circuit Diagram")

    # Save the diagram
    save_path = os.path.join(results_dir,"circuit_visuals", f"_circuit_{dname}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved circuit diagram at {save_path}")

# ===== Definition of the Huber loss and metrics =====
def huber_loss(y_true, y_pred, delta=delta_huber):
    error = y_true - y_pred
    is_small_error = torch.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(is_small_error, squared_loss, linear_loss)

def loss_fn(circuit, params, X, Y, delta=delta_huber):
    # Calculate the average loss over the entire dataset
    preds = torch.stack([circuit(params, x) for x in X])
    return torch.mean(huber_loss(Y, preds, delta))

def accuracy_fn(circuit, params, X, Y, threshold=accuracy_threshold):
    preds = torch.stack([circuit(params, x) for x in X])
    correct = torch.sum(torch.abs(preds - Y) < threshold)
    return correct.item() / len(Y)

# ===== Initialization of parameters =====
# For the QRU circuit: 2 parameters per input per layer -> (depth, 2*n_input)
params_qru = torch.tensor(np.full((depth, 2 * n_input), 0.5),
                          requires_grad=True, dtype=torch.float64)
# For the PQC circuit: 2 parameters per input per layer -> (depth, 2*n_input)
params_pqc = torch.tensor(np.full((depth, 2 * n_input), 0.5),
                          requires_grad=True, dtype=torch.float64)

# Display summary of both circuits (number of trainable parameters)
print("QRU - Number of trainable parameters:", params_qru.numel())
print("PQC - Number of trainable parameters:", params_pqc.numel())

# Separate optimizers
optimizer_qru = torch.optim.Adam([params_qru], lr=learning_rate)
optimizer_pqc = torch.optim.Adam([params_pqc], lr=learning_rate)

# Draw the circuit visualization
sample_input = torch.rand(n_input, dtype=torch.float64)
print_model_summary(circuit_qru, sample_input, "QRU")

# Draw the circuit visualization
sample_input = torch.rand(n_input, dtype=torch.float64)
print_model_summary(circuit_pqc, sample_input, "PQC")

# ===== Saving metrics =====
# For QRU
train_loss_qru, test_loss_qru = [], []
train_acc_qru, test_acc_qru = [], []
grad_norm_qru = []
train_preds_qru, test_preds_qru = [], []
# For PQC
train_loss_pqc, test_loss_pqc = [], []
train_acc_pqc, test_acc_pqc = [], []
grad_norm_pqc = []
train_preds_pqc, test_preds_pqc = [], []

log_file_path = os.path.join(results_dir, "metrics_comparison.txt")
log_file = open(log_file_path, "w")
log_file.write("epoch, train_loss_qru, test_loss_qru, train_acc_qru, test_acc_qru, grad_norm_qru, "
               "train_loss_pqc, test_loss_pqc, train_acc_pqc, test_acc_pqc, grad_norm_pqc\n")
log_file.flush()

# ===== Joint training loop =====
for epoch in range(num_epochs):
    # --- Update the QRU circuit ---
    optimizer_qru.zero_grad()
    loss_qru = loss_fn(circuit_qru, params_qru, X_train, Y_train)
    loss_qru.backward()
    grad_norm_val_qru = params_qru.grad.norm().item() if params_qru.grad is not None else 0
    optimizer_qru.step()

    # --- Update the PQC circuit ---
    optimizer_pqc.zero_grad()
    loss_pqc = loss_fn(circuit_pqc, params_pqc, X_train, Y_train)
    loss_pqc.backward()
    grad_norm_val_pqc = params_pqc.grad.norm().item() if params_pqc.grad is not None else 0
    optimizer_pqc.step()

    # Evaluate metrics (no gradient)
    current_train_loss_qru = loss_fn(circuit_qru, params_qru, X_train, Y_train).item()
    current_test_loss_qru = loss_fn(circuit_qru, params_qru, X_test, Y_test).item()
    current_train_acc_qru = accuracy_fn(circuit_qru, params_qru, X_train, Y_train)
    current_test_acc_qru = accuracy_fn(circuit_qru, params_qru, X_test, Y_test)
    current_train_loss_pqc = loss_fn(circuit_pqc, params_pqc, X_train, Y_train).item()
    current_test_loss_pqc = loss_fn(circuit_pqc, params_pqc, X_test, Y_test).item()
    current_train_acc_pqc = accuracy_fn(circuit_pqc, params_pqc, X_train, Y_train)
    current_test_acc_pqc = accuracy_fn(circuit_pqc, params_pqc, X_test, Y_test)

    # Predictions for visualization
    preds_train_qru = torch.stack([circuit_qru(params_qru, x) for x in X_train]).detach().numpy()
    preds_test_qru = torch.stack([circuit_qru(params_qru, x) for x in X_test]).detach().numpy()
    preds_train_pqc = torch.stack([circuit_pqc(params_pqc, x) for x in X_train]).detach().numpy()
    preds_test_pqc = torch.stack([circuit_pqc(params_pqc, x) for x in X_test]).detach().numpy()

    train_loss_qru.append(current_train_loss_qru)
    test_loss_qru.append(current_test_loss_qru)
    train_acc_qru.append(current_train_acc_qru)
    test_acc_qru.append(current_test_acc_qru)
    grad_norm_qru.append(grad_norm_val_qru)
    train_preds_qru.append(preds_train_qru)
    test_preds_qru.append(preds_test_qru)

    train_loss_pqc.append(current_train_loss_pqc)
    test_loss_pqc.append(current_test_loss_pqc)
    train_acc_pqc.append(current_train_acc_pqc)
    test_acc_pqc.append(current_test_acc_pqc)
    grad_norm_pqc.append(grad_norm_val_pqc)
    train_preds_pqc.append(preds_train_pqc)
    test_preds_pqc.append(preds_test_pqc)

    df_train_pred_pqc = pd.DataFrame({
    "Y_train": Y_train.detach().numpy(),
    "train_preds_pqc": train_preds_pqc[-1],
    })
    df_test_pred_pqc = pd.DataFrame({
    "Y_test": Y_test.detach().numpy(),
    "test_preds_pqc": test_preds_pqc[-1]
    })

    df_train_pred_qru = pd.DataFrame({
    "Y_train": Y_train.detach().numpy(),
    "train_preds_qru": train_preds_qru[-1],
    })
    df_test_pred_qru = pd.DataFrame({
    "Y_test": Y_test.detach().numpy(),
    "test_preds_qru": test_preds_qru[-1]
    })


    log_line = (f"{epoch}, {current_train_loss_qru:.6f}, {current_test_loss_qru:.6f}, "
                f"{current_train_acc_qru:.6f}, {current_test_acc_qru:.6f}, {grad_norm_val_qru:.6f}, "
                f"{current_train_loss_pqc:.6f}, {current_test_loss_pqc:.6f}, "
                f"{current_train_acc_pqc:.6f}, {current_test_acc_pqc:.6f}, {grad_norm_val_pqc:.6f}\n")
    log_file.write(log_line)
    log_file.flush()

    print(f"Epoch {epoch}: QRU -> Loss: {current_train_loss_qru:.4f}, Test Loss: {current_test_loss_qru:.4f}, "
          f"Train Acc: {current_train_acc_qru:.4f}, Test Acc: {current_test_acc_qru:.4f}, Grad Norm: {grad_norm_val_qru:.4f} | "
          f"PQC -> Loss: {current_train_loss_pqc:.4f}, Test Loss: {current_test_loss_pqc:.4f}, "
          f"Train Acc: {current_train_acc_pqc:.4f}, Test Acc: {current_test_acc_pqc:.4f}, Grad Norm: {grad_norm_val_pqc:.4f}")

log_file.close()

df_train_pred_pqc.to_csv("MG_PQC_vs_QRU_15.03/PQC_train_predictions.csv", index=False)
df_test_pred_pqc.to_csv("MG_PQC_vs_QRU_15.03/PQC_test_predictions.csv", index=False)
df_train_pred_qru.to_csv("MG_PQC_vs_QRU_15.03/QRU_train_predictions.csv", index=False)
df_test_pred_qru.to_csv("MG_PQC_vs_QRU_15.03/QRU_test_predictions.csv", index=False)

# ===== Save Trained Parameters =====
torch.save(params_qru.detach(), os.path.join(results_dir, "trained_params_qru.pt"))
torch.save(params_pqc.detach(), os.path.join(results_dir, "trained_params_pqc.pt"))
print("Trained parameters saved.")


epochs = range(num_epochs)

# ===== Saving figures =====
# Loss figure
plt.figure()
plt.plot(epochs, train_loss_qru, label="Train Loss QRU")
plt.plot(epochs, test_loss_qru, label="Test Loss QRU")
plt.plot(epochs, train_loss_pqc, label="Train Loss PQC")
plt.plot(epochs, test_loss_pqc, label="Test Loss PQC")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs (Mackey Glass)")
plt.legend()
plt.savefig(os.path.join(results_dir, "loss_comparison.png"), dpi=300)
plt.close()

# Accuracy figure
plt.figure()
plt.plot(epochs, train_acc_qru, label="Train Acc QRU")
plt.plot(epochs, test_acc_qru, label="Test Acc QRU")
plt.plot(epochs, train_acc_pqc, label="Train Acc PQC")
plt.plot(epochs, test_acc_pqc, label="Test Acc PQC")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs (Mackey Glass)")
plt.legend()
plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"), dpi=300)
plt.close()

# Gradient Norm figure
plt.figure()
plt.plot(epochs, grad_norm_qru, label="Grad Norm QRU")
plt.plot(epochs, grad_norm_pqc, label="Grad Norm PQC")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm over Epochs (Mackey Glass)")
plt.legend()
plt.savefig(os.path.join(results_dir, "grad_norm_comparison.png"), dpi=300)
plt.close()

# Train: Predictions vs Actual Values figure (QRU vs PQC)
plt.figure()
plt.plot(train_preds_qru[-1], 'r-', label="QRU Predictions")
plt.plot(train_preds_pqc[-1], 'g-', label="PQC Predictions")
plt.plot(Y_train.detach().numpy(), 'b--', label="Actual Values")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.title("Train: Predictions vs Actual Values (QRU vs PQC)")
plt.legend()
plt.savefig(os.path.join(results_dir, "train_predictions_comparison.png"), dpi=300)
plt.close()

# Test: Predictions vs Actual Values figure (QRU vs PQC)
plt.figure()
plt.plot(test_preds_qru[-1], 'r-', label="QRU Predictions")
plt.plot(test_preds_pqc[-1], 'g-', label="PQC Predictions")
plt.plot(Y_test.detach().numpy(), 'b--', label="Actual Values")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.title("Test: Predictions vs Actual Values (QRU vs PQC)")
plt.legend()
plt.savefig(os.path.join(results_dir, "test_predictions_comparison.png"), dpi=300)
plt.close()

print("Training complete. Metrics and figures have been saved in the directory", results_dir)