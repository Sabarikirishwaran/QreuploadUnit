import os
import torch
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ===== Parameters =====
depth = 3         # Depth (number of layers) for both models
n_input = 3       # Number of input points (window size)
total_points = 500  # Total simulated points for Mackey Glass
train_ratio = 0.7
num_epochs = 500
learning_rate = 0.01
delta_huber = 0.1         # Huber loss parameter
accuracy_threshold = 0.1  # Threshold for a "correct" prediction

# Create directory to save results
results_dir = "MG_QRU_vs_RNN"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ===== Mackey Glass Series Generation =====
def generate_mackey_glass(total_steps, tau=17, n=10, beta=0.2, gamma=0.1, dt=1):
    """
    Simulate the Mackey Glass time series using Euler's method.
    Initial conditions (for t < tau) are set to 1.2.
    """
    history = np.zeros(total_steps + tau)
    history[:tau] = 1.2
    for t in range(tau, total_steps + tau):
        history[t] = history[t-1] + dt * (beta * history[t-tau] / (1 + history[t-tau]**n) - gamma * history[t-1])
    return history[tau:]

mackey_series = generate_mackey_glass(total_points)

# ===== Normalize the series between -1 and 1 =====
min_val = np.min(mackey_series)
max_val = np.max(mackey_series)
mackey_series_norm = 2 * (mackey_series - min_val) / (max_val - min_val) - 1

# ===== Create Sliding Windows =====
def create_windows(series, window_size):
    X = []
    Y = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        Y.append(series[i+window_size])
    return np.array(X), np.array(Y)

X, Y = create_windows(mackey_series_norm, n_input)

# Convert to torch tensors (using float64 to be consistent with PennyLane)
X = torch.tensor(X, dtype=torch.float64)
Y = torch.tensor(Y, dtype=torch.float64)

# Split data into training and testing sets
split_idx = int(len(X) * train_ratio)
X_train, Y_train = X[:split_idx], Y[:split_idx]
X_test, Y_test = X[split_idx:], Y[split_idx:]

# ===== Define the Quantum Circuit (QRU) =====
dev = qml.device("default.qubit", wires=1)

def QRU(params, x, alpha=0.5):
    # Quantum circuit with re-uploading using 3 rotations per input element
    for i in range(depth):
        for j in range(len(x)):
            qml.RX(params[i][3 * j], wires=0)
            qml.RY(params[i][3 * j + 1] * x[j], wires=0)
            qml.RZ(params[i][3 * j + 2], wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="torch")
def quantum_circuit(params, x):
    return QRU(params, x)

# ===== Define the Classical RNN =====
def classical_RNN(params, x):
    """
    A simple classical RNN with the same number of parameters as the quantum model.
    It processes the input sequence and updates a scalar hidden state.
    params: tensor of shape (depth, 3*n_input)
    x: input sequence of length n_input
    """
    h = torch.tensor(0.0, dtype=torch.float64)
    for i in range(depth):
        for j in range(len(x)):
            a = params[i][3 * j]
            b = params[i][3 * j + 1]
            c = params[i][3 * j + 2]
            h = torch.tanh(a * h + b * x[j] + c)
    return h

# ===== Define Loss Function and Metrics =====
def huber_loss(y_true, y_pred, delta=delta_huber):
    error = y_true - y_pred
    is_small_error = torch.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(is_small_error, squared_loss, linear_loss)

def loss_fn(model_fn, params, X, Y, delta=delta_huber):
    # Compute mean loss over the dataset using the provided model function
    preds = torch.stack([model_fn(params, x) for x in X])
    return torch.mean(huber_loss(Y, preds, delta))

def accuracy_fn(model_fn, params, X, Y, threshold=accuracy_threshold):
    preds = torch.stack([model_fn(params, x) for x in X])
    correct = torch.sum(torch.abs(preds - Y) < threshold)
    return correct.item() / len(Y)

# ===== Initialize Parameters for both models =====
# Both models have exactly depth * 3 * n_input parameters.
initial_params = np.full((depth, 3 * n_input), 0.5)
quantum_params = torch.tensor(initial_params, requires_grad=True, dtype=torch.float64)
classical_params = torch.tensor(initial_params, requires_grad=True, dtype=torch.float64)

# Use Adam optimizer for both models
quantum_optimizer = torch.optim.Adam([quantum_params], lr=learning_rate)
classical_optimizer = torch.optim.Adam([classical_params], lr=learning_rate)

# ===== Lists to Store Metrics =====
# Quantum model metrics
q_train_loss_list = []
q_test_loss_list = []
q_train_acc_list = []
q_test_acc_list = []
q_grad_norm_list = []
q_train_preds_list = None
q_test_preds_list = None

# Classical model metrics
r_train_loss_list = []
r_test_loss_list = []
r_train_acc_list = []
r_test_acc_list = []
r_grad_norm_list = []
r_train_preds_list = None
r_test_preds_list = None

# ===== CSV Logging =====
csv_file_path = os.path.join(results_dir, "metrics.csv")
csv_file = open(csv_file_path, "w")
csv_file.write("epoch,q_train_loss,q_test_loss,q_train_acc,q_test_acc,q_grad_norm,"
               "r_train_loss,r_test_loss,r_train_acc,r_test_acc,r_grad_norm\n")
csv_file.flush()

# ===== Training Loop =====
for epoch in range(num_epochs):
    # ---- Quantum Model Training Step ----
    quantum_optimizer.zero_grad()
    q_loss = loss_fn(quantum_circuit, quantum_params, X_train, Y_train)
    q_loss.backward()
    q_grad_norm = quantum_params.grad.norm().item() if quantum_params.grad is not None else 0
    quantum_optimizer.step()
    
    # Evaluate quantum model metrics
    q_train_loss = loss_fn(quantum_circuit, quantum_params, X_train, Y_train).item()
    q_test_loss = loss_fn(quantum_circuit, quantum_params, X_test, Y_test).item()
    q_train_acc = accuracy_fn(quantum_circuit, quantum_params, X_train, Y_train)
    q_test_acc = accuracy_fn(quantum_circuit, quantum_params, X_test, Y_test)
    q_train_preds = torch.stack([quantum_circuit(quantum_params, x) for x in X_train]).detach().numpy()
    q_test_preds = torch.stack([quantum_circuit(quantum_params, x) for x in X_test]).detach().numpy()
    
    q_train_loss_list.append(q_train_loss)
    q_test_loss_list.append(q_test_loss)
    q_train_acc_list.append(q_train_acc)
    q_test_acc_list.append(q_test_acc)
    q_grad_norm_list.append(q_grad_norm)
    q_train_preds_list = q_train_preds  # update latest predictions
    q_test_preds_list = q_test_preds

    # ---- Classical RNN Training Step ----
    classical_optimizer.zero_grad()
    r_loss = loss_fn(classical_RNN, classical_params, X_train, Y_train)
    r_loss.backward()
    r_grad_norm = classical_params.grad.norm().item() if classical_params.grad is not None else 0
    classical_optimizer.step()
    
    # Evaluate classical model metrics
    r_train_loss = loss_fn(classical_RNN, classical_params, X_train, Y_train).item()
    r_test_loss = loss_fn(classical_RNN, classical_params, X_test, Y_test).item()
    r_train_acc = accuracy_fn(classical_RNN, classical_params, X_train, Y_train)
    r_test_acc = accuracy_fn(classical_RNN, classical_params, X_test, Y_test)
    r_train_preds = torch.stack([classical_RNN(classical_params, x) for x in X_train]).detach().numpy()
    r_test_preds = torch.stack([classical_RNN(classical_params, x) for x in X_test]).detach().numpy()
    
    r_train_loss_list.append(r_train_loss)
    r_test_loss_list.append(r_test_loss)
    r_train_acc_list.append(r_train_acc)
    r_test_acc_list.append(r_test_acc)
    r_grad_norm_list.append(r_grad_norm)
    r_train_preds_list = r_train_preds  # update latest predictions
    r_test_preds_list = r_test_preds
    
    # Log metrics to CSV (flush each epoch)
    log_line = f"{epoch},{q_train_loss:.6f},{q_test_loss:.6f},{q_train_acc:.6f},{q_test_acc:.6f},{q_grad_norm:.6f}," \
               f"{r_train_loss:.6f},{r_test_loss:.6f},{r_train_acc:.6f},{r_test_acc:.6f},{r_grad_norm:.6f}\n"
    csv_file.write(log_line)
    csv_file.flush()
    
    print(f"Epoch {epoch}: Quantum -> Train Loss: {q_train_loss:.4f}, Test Loss: {q_test_loss:.4f}, Train Acc: {q_train_acc:.4f}, Test Acc: {q_test_acc:.4f}, Grad Norm: {q_grad_norm:.4f} | "
          f"Classical -> Train Loss: {r_train_loss:.4f}, Test Loss: {r_test_loss:.4f}, Train Acc: {r_train_acc:.4f}, Test Acc: {r_test_acc:.4f}, Grad Norm: {r_grad_norm:.4f}")

csv_file.close()

# ===== Plot and Save Figures =====
epochs = range(num_epochs)

# Loss Plot
plt.figure()
plt.plot(epochs, q_train_loss_list, label="QRU Train")
plt.plot(epochs, q_test_loss_list, label="QRU Test")
plt.plot(epochs, r_train_loss_list, label="RNN Train")
plt.plot(epochs, r_test_loss_list, label="RNN Test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.savefig(os.path.join(results_dir, "loss.png"), dpi=300)
plt.close()

# Accuracy Plot
plt.figure()
plt.plot(epochs, q_train_acc_list, label="QRU Train")
plt.plot(epochs, q_test_acc_list, label="QRU Test")
plt.plot(epochs, r_train_acc_list, label="RNN Train")
plt.plot(epochs, r_test_acc_list, label="RNN Test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.savefig(os.path.join(results_dir, "accuracy.png"), dpi=300)
plt.close()

# Gradient Norm Plot
plt.figure()
plt.plot(epochs, q_grad_norm_list, label="QRU")
plt.plot(epochs, r_grad_norm_list, label="RNN")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm over Epochs")
plt.legend()
plt.savefig(os.path.join(results_dir, "grad_norm.png"), dpi=300)
plt.close()

# Train Predictions Plot
plt.figure()
plt.plot(q_train_preds_list, 'r-', label="QRU")
plt.plot(r_train_preds_list, 'g-', label="RNN")
plt.plot(Y_train.detach().numpy(), 'b--', label="Actual Values")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.title("Train Set: Predictions vs Actual Values")
plt.legend()
plt.savefig(os.path.join(results_dir, "train_predictions.png"), dpi=300)
plt.close()

# Test Predictions Plot
plt.figure()
plt.plot(q_test_preds_list, 'r-', label="QRU")
plt.plot(r_test_preds_list, 'g-', label="RNN")
plt.plot(Y_test.detach().numpy(), 'b--', label="Actual Values")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.title("Test Set: Predictions vs Actual Values")
plt.legend()
plt.savefig(os.path.join(results_dir, "test_predictions.png"), dpi=300)
plt.close()

print("Training completed. Metrics and figures have been saved in the directory", results_dir)