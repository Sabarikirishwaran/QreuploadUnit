import argparse
import os
import random
import pandas as pd
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
from tqdm import trange

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

nb_reuploading = 3
seq_length = 3
nb_qubit_reupload = 1
nb_qubit_pqc = 1
nb_qubit_vqc = 5
L_vqc = 1
num_variational = 3
nb_epoch = 300
lr = 0.01
depth_pqc = 5

dev_reupload = qml.device("default.qubit", wires=nb_qubit_reupload, shots=None)
dev_vqc = qml.device("default.qubit", wires=nb_qubit_vqc, shots=None)
dev_pqc = qml.device("default.qubit", wires=1, shots=None)

def encoding_layer(params_encoding, x, seq_length, nb_qubit):
    rotation_gates = [qml.RX, qml.RY, qml.RZ]
    for j in range(seq_length):
        gate = rotation_gates[j % len(rotation_gates)]
        target = j if j < nb_qubit else j % nb_qubit
        gate(params_encoding[j] * x[j], wires=target)

def variational_layer(params_variational, seq_length, nb_qubit, num_variational):
    rotation_gates = [qml.RX, qml.RY, qml.RZ]
    for q in range(nb_qubit):
        for k in range(num_variational):
            gate = rotation_gates[k % len(rotation_gates)]
            idx = q * num_variational + k
            gate(params_variational[idx], wires=q)

@qml.qnode(dev_reupload, interface="torch")
def quantum_circuit_reupload(params, x):
    params_enc, params_var = params
    for i in range(nb_reuploading):
        encoding_layer(params_enc[i], x, seq_length, nb_qubit_reupload)
        variational_layer(params_var[i], seq_length, nb_qubit_reupload, num_variational)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev_vqc, interface="torch")
def quantum_circuit_vqc_bague(params, x):
    params_enc, params_var = params
    encoding_layer(params_enc, x, seq_length, nb_qubit_vqc)
    for i in range(L_vqc):
        variational_layer(params_var[i], seq_length, nb_qubit_vqc, num_variational)
        for j in range(nb_qubit_vqc):
            qml.CNOT(wires=[j, (j + 1) % nb_qubit_vqc])
    for q in reversed(range(nb_qubit_vqc - 1)):
        qml.CNOT(wires=[q + 1, q])
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev_pqc, interface="torch")
def quantum_circuit_pqc(params, x):
    params_enc, params_var = params
    encoding_layer(params_enc, x, seq_length, nb_qubit_pqc)
    for i in range(depth_pqc):
        variational_layer(params_var[i], seq_length, nb_qubit_pqc, num_variational)
    return qml.expval(qml.PauliZ(0))

def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

def prediction_accuracy(y_pred, y_true, tolerance=0.1):
    correct = torch.sum(torch.abs(y_pred - y_true) < tolerance)
    return 100 * correct / len(y_true)

def preprocess_river_data(file_path):
    df = pd.read_csv(file_path, skiprows=2)
    df.columns = ['date', 'wlvalue', 'fvalue']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df.set_index('date', inplace=True)
    df['wlvalue_normalized'] = 2 * (df['wlvalue'] - df['wlvalue'].min()) / (df['wlvalue'].max() - df['wlvalue'].min()) - 1
    return df['wlvalue_normalized'].values

def preprocess_mg(series):
    norm = (series - series.min()) / (series.max() - series.min())
    return 2 * norm - 1

def generate_mackey_glass(n_points=2000, tau=17, dt=1, initial_value=1.2, beta=0.2, gamma=0.1, n=10):
    hist = int(tau)
    total = n_points + hist
    s = np.zeros(total)
    s[:hist] = initial_value
    for t in range(hist, total):
        s[t] = s[t-1] + dt * (
            beta * s[t-hist] / (1 + s[t-hist]**n)
            - gamma * s[t-1]
        )
    return s[hist:]

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def split_data(data, test_ratio, seq_length):
    X, y = create_sequences(data, seq_length)
    split = int(len(X) * (1 - test_ratio))
    return X[:split], y[:split], X[split:], y[split:]

def plot_metrics(loss_hist, acc_hist, grad_norm_hist, name):
    epochs = range(1, len(loss_hist)+1)
    plt.figure()
    plt.plot(epochs, loss_hist, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{name}: Test Loss over Epochs")
    plt.legend()
    plt.savefig(f"{name}_loss_plot.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, acc_hist, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{name}: Test Accuracy over Epochs")
    plt.legend()
    plt.savefig(f"{name}_accuracy_plot.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, grad_norm_hist, label="Gradient Norm")
    plt.xlabel("Epochs")
    plt.ylabel("L2 Norm")
    plt.title(f"{name}: Gradient Norm over Epochs")
    plt.legend()
    plt.savefig(f"{name}_gradnorm_plot.png", dpi=300)
    plt.close()

def plot_predictions(y_true, y_pred, name):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Ground Truth", linewidth=2)
    plt.plot(y_pred, label="Predictions", linestyle="--")
    plt.xlabel("Index")
    plt.ylabel("Normalized Value")
    plt.title(f"{name}: Predictions vs Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{name}_prediction_plot.png", dpi=300)
    plt.close()

def train_and_eval(model, params_init, X_tr, y_tr, X_te, y_te, name, epochs, lr):
    params = [p.clone().detach().requires_grad_(True) for p in params_init]
    opt = torch.optim.RMSprop(params, lr=lr)
    loss_hist, acc_hist, grad_norm_hist = [], [], []
    param_count = sum(p.numel() for p in params if p.requires_grad)
    print(f"Trainable parameters: {param_count}")

    with open(f"{name}_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "test_loss", "test_acc", "grad_norm"])

        for ep in trange(epochs, desc=f"Training {name}"):
            opt.zero_grad()
            preds = torch.stack([model(params, x) for x in X_tr])
            loss = mse_loss(preds, y_tr)
            loss.backward()
            grad_norm = torch.sqrt(sum(p.grad.data.norm(2).item() ** 2 for p in params if p.grad is not None))
            opt.step()

            with torch.no_grad():
                preds_t = torch.stack([model(params, x) for x in X_te])
                loss_t = mse_loss(preds_t, y_te).item()
                acc_t = prediction_accuracy(preds_t, y_te).item()

            loss_hist.append(loss_t)
            acc_hist.append(acc_t)
            grad_norm_hist.append(grad_norm)

            writer.writerow([ep+1, loss_t, acc_t, grad_norm])
            f.flush()

    plot_metrics(loss_hist, acc_hist, grad_norm_hist, name)
    torch.save([p.detach() for p in params], f"{name}_trained_params.pt")

    with torch.no_grad():
        preds_final = torch.stack([model(params, x) for x in X_te])
        with open(f"{name}_predictions.csv", "w", newline="") as f_pred:
            writer = csv.writer(f_pred)
            writer.writerow(["Index", "Actual", "Predicted"])
            for i, (true_val, pred_val) in enumerate(zip(y_te, preds_final)):
                writer.writerow([i, true_val.item(), pred_val.item()])
        plot_predictions(y_te.numpy(), preds_final.numpy(), name)    

def main(selected_model, data):
    X_tr_np, y_tr_np, X_te_np, y_te_np = split_data(data, 0.2, seq_length)
    X_tr = torch.tensor(X_tr_np, dtype=torch.float64)
    y_tr = torch.tensor(y_tr_np, dtype=torch.float64)
    X_te = torch.tensor(X_te_np, dtype=torch.float64)
    y_te = torch.tensor(y_te_np, dtype=torch.float64)

    params_init_reupload = [
        torch.full((nb_reuploading, seq_length), np.pi, dtype=torch.float64, requires_grad=True),
        torch.full((nb_reuploading, num_variational), np.pi, dtype=torch.float64, requires_grad=True)
    ]

    params_init_vqc_bague = [
        torch.full((seq_length,), np.pi, dtype=torch.float64, requires_grad=True),
        torch.full((L_vqc, nb_qubit_vqc * num_variational), np.pi, dtype=torch.float64, requires_grad=True)
    ]

    params_init_pqc = [
        torch.full((seq_length,), np.pi, dtype=torch.float64, requires_grad=True),
        torch.full((depth_pqc, num_variational), np.pi, dtype=torch.float64, requires_grad=True)
    ]

    if selected_model in ("QRU", "all"):
        train_and_eval(quantum_circuit_reupload, params_init_reupload, X_tr, y_tr, X_te, y_te, "QRU_monoqubit", nb_epoch, lr)

    if selected_model in ("VQC_bague", "all"):
        train_and_eval(quantum_circuit_vqc_bague, params_init_vqc_bague, X_tr, y_tr, X_te, y_te, "VQC_bague_inverse", nb_epoch, lr)

    if selected_model in ("PQC_simple", "all"):
        train_and_eval(quantum_circuit_pqc, params_init_pqc, X_tr, y_tr, X_te, y_te, "PQC_simple", nb_epoch, lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QRU/VQC/PQC models on River or MackeyGlass data")
    parser.add_argument("--model", choices=["QRU", "VQC_bague", "PQC_simple", "all"], default="QRU", help="Which model to run")
    parser.add_argument("--dataset", choices=["river", "mackey"], default="mackey", help="Dataset to train on")
    args = parser.parse_args()

    river_data_path = "data/river_level.csv"

    if args.dataset == "river":
        if not os.path.exists(river_data_path):
            raise FileNotFoundError("River level CSV not found. Please place it in 'data/river_level.csv'")
        data = preprocess_river_data(river_data_path)
    else:
        series = generate_mackey_glass()
        data = preprocess_mg(series)

    data = torch.tensor(data, dtype=torch.float64)
    main(args.model, data)
