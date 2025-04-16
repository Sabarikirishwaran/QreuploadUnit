import os
import torch
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Pour la reproductibilité
np.random.seed(42)
torch.manual_seed(42)

# ===== Paramètres =====
depth = 3         # Profondeur du circuit
n_input = 3       # Nombre de points d'entrée (taille de la fenêtre)
total_points = 300  # Nombre total de points simulés pour Mackey Glass
train_ratio = 0.7
num_epochs = 300
learning_rate = 0.01
delta_huber = 0.1         # Paramètre de la Huber loss
accuracy_threshold = 0.1  # Seuil pour définir une prédiction "correcte"

# Création du répertoire pour enregistrer les résultats
results_dir = "River_QRU_vs_QRB_14.03"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ===== Génération de la série de Mackey Glass =====
def preprocess_river_data(file_path):
    """
    Preprocess the data by keeping only wlvalue and normalizing it between -1 and 1.
    """
    df = pd.read_csv(file_path, skiprows=2)
    df.columns = ['date', 'wlvalue', 'fvalue']
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['date'])
    df.set_index('date', inplace=True)
    # Normalize between -1 and 1
    df['wlvalue_normalized'] = 2*(df['wlvalue'] - df['wlvalue'].min()) / (df['wlvalue'].max() - df['wlvalue'].min()) - 1
    return df[['wlvalue_normalized']]

def create_sequences(data, n_input=3):
    """
    Create sequences based on wlvalue.
    """
    X, y = [], []
    for i in range(len(data) - n_input):
        wl_seq = data.iloc[i:i + n_input, 0].values
        X.append(wl_seq)
        y.append(data.iloc[i + n_input, 0])
    return np.array(X), np.array(y)

def split_data(data, test_size=0.2, n_input=3):
    """
    Split the data into training and test sets.
    """
    X, y = create_sequences(data, n_input=n_input)
    train_size = int((1 - test_size) * len(X))
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

file_path = "QLSTM-DRU-Environmental-TimeSeries\\QRU+QRBs\\river_level.csv"
data = preprocess_river_data(file_path)
X_train_np, y_train_np, X_test_np, y_test_np = split_data(data, n_input=n_input)
X_train = torch.tensor(X_train_np, dtype=torch.float64)
Y_train = torch.tensor(y_train_np, dtype=torch.float64)
X_test  = torch.tensor(X_test_np, dtype=torch.float64)
Y_test  = torch.tensor(y_test_np, dtype=torch.float64)

# ===== Configuration du device =====
# On utilise 2 fils pour pouvoir simuler le circuit QRU_QRB qui opère sur wires 0 et 1
dev = qml.device("default.qubit", wires=2)

# ===== Définition des circuits =====
def QRU(params, x, alpha=0.5):
    # Circuit à re-uploading avec 3 rotations par entrée (ordre modifié : RY, RX, RZ)
    for i in range(depth):
        for j in range(len(x)):
            qml.RY(params[i][3 * j + 1] * x[j], wires=0)
            qml.RX(params[i][3 * j], wires=0)
            qml.RZ(params[i][3 * j + 2], wires=0)
    return qml.expval(qml.PauliZ(0))

def QRU_QRB(params, x, alpha=0.5):
    # Circuit avec re-uploading sur wire 0 et utilisation d'une seconde wire (wire 1)
    qml.Hadamard(wires=1)
    for i in range(depth):
        for j in range(len(x)):
            qml.CRY(params[i][3 * j + 1] * x[j], wires=[1, 0])
        #qml.CRX(alpha * params[i][3 * len(x) + 1], wires=[1, 0])    
        for j in range(len(x)):
            qml.RX(params[i][3 * j], wires=0)
            qml.RZ(params[i][3 * j + 2], wires=0)
    qml.Hadamard(wires=1)
    proj = (qml.Identity(wires=1) + qml.PauliZ(wires=1)) / 2
    return qml.expval(qml.PauliZ(0) @ proj)

@qml.qnode(dev, interface="torch")
def circuit_qru(params, x):
    return QRU(params, x)

@qml.qnode(dev, interface="torch")
def circuit_qru_qrb(params, x):
    return QRU_QRB(params, x)

# ===== Définition de la Huber loss et des métriques =====
def huber_loss(y_true, y_pred, delta=delta_huber):
    error = y_true - y_pred
    is_small_error = torch.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(is_small_error, squared_loss, linear_loss)

def loss_fn(circuit, params, X, Y, delta=delta_huber):
    # Calcule la loss moyenne sur l'ensemble des données pour le circuit passé en argument
    preds = torch.stack([circuit(params, x) for x in X])
    return torch.mean(huber_loss(Y, preds, delta))

def accuracy_fn(circuit, params, X, Y, threshold=accuracy_threshold):
    preds = torch.stack([circuit(params, x) for x in X])
    correct = torch.sum(torch.abs(preds - Y) < threshold)
    return correct.item() / len(Y)

# ===== Initialisation des paramètres =====
# Pour QRU : 3 paramètres par entrée -> shape: (depth, 3*n_input)
params_qru = torch.tensor(np.full((depth, 3 * n_input), 0.5),
                          requires_grad=True, dtype=torch.float64)
# Pour QRU_QRB : 3*n_input + 1 paramètres par couche (le +1 pour la porte CRX)
params_qru_qrb = torch.tensor(np.full((depth, 3 * n_input + 2), 0.5),
                              requires_grad=True, dtype=torch.float64)

# Optimiseurs distincts
optimizer_qru = torch.optim.Adam([params_qru], lr=learning_rate)
optimizer_qru_qrb = torch.optim.Adam([params_qru_qrb], lr=learning_rate)

# ===== Sauvegarde des métriques =====
train_loss_qru_list = []
test_loss_qru_list = []
train_acc_qru_list = []
test_acc_qru_list = []
grad_norm_qru_list = []

train_loss_qru_qrb_list = []
test_loss_qru_qrb_list = []
train_acc_qru_qrb_list = []
test_acc_qru_qrb_list = []
grad_norm_qru_qrb_list = []

train_preds_qru_list = []
test_preds_qru_list = []
train_preds_qru_qrb_list = []
test_preds_qru_qrb_list = []

log_file_path = os.path.join(results_dir, "metrics_comparison.txt")
log_file = open(log_file_path, "w")
log_file.write("epoch, train_loss_qru, test_loss_qru, train_acc_qru, test_acc_qru, grad_norm_qru, "
               "train_loss_qru_qrb, test_loss_qru_qrb, train_acc_qru_qrb, test_acc_qru_qrb, grad_norm_qru_qrb\n")
log_file.flush()



# ===== Boucle d'entraînement =====
for epoch in range(num_epochs):
    # Mise à jour du circuit QRU
    optimizer_qru.zero_grad()
    loss_qru = loss_fn(circuit_qru, params_qru, X_train, Y_train)
    loss_qru.backward()
    grad_norm_qru = params_qru.grad.norm().item() if params_qru.grad is not None else 0
    optimizer_qru.step()
    
    # Mise à jour du circuit QRU_QRB
    optimizer_qru_qrb.zero_grad()
    loss_qru_qrb = loss_fn(circuit_qru_qrb, params_qru_qrb, X_train, Y_train)
    loss_qru_qrb.backward()
    grad_norm_qru_qrb = params_qru_qrb.grad.norm().item() if params_qru_qrb.grad is not None else 0
    optimizer_qru_qrb.step()
    
    # Évaluation des métriques (sans gradient)
    train_loss_qru = loss_fn(circuit_qru, params_qru, X_train, Y_train).item()
    test_loss_qru = loss_fn(circuit_qru, params_qru, X_test, Y_test).item()
    train_acc_qru = accuracy_fn(circuit_qru, params_qru, X_train, Y_train)
    test_acc_qru = accuracy_fn(circuit_qru, params_qru, X_test, Y_test)
    
    train_loss_qru_qrb = loss_fn(circuit_qru_qrb, params_qru_qrb, X_train, Y_train).item()
    test_loss_qru_qrb = loss_fn(circuit_qru_qrb, params_qru_qrb, X_test, Y_test).item()
    train_acc_qru_qrb = accuracy_fn(circuit_qru_qrb, params_qru_qrb, X_train, Y_train)
    test_acc_qru_qrb = accuracy_fn(circuit_qru_qrb, params_qru_qrb, X_test, Y_test)
    
    train_loss_qru_list.append(train_loss_qru)
    test_loss_qru_list.append(test_loss_qru)
    train_acc_qru_list.append(train_acc_qru)
    test_acc_qru_list.append(test_acc_qru)
    grad_norm_qru_list.append(grad_norm_qru)
    
    train_loss_qru_qrb_list.append(train_loss_qru_qrb)
    test_loss_qru_qrb_list.append(test_loss_qru_qrb)
    train_acc_qru_qrb_list.append(train_acc_qru_qrb)
    test_acc_qru_qrb_list.append(test_acc_qru_qrb)
    grad_norm_qru_qrb_list.append(grad_norm_qru_qrb)
    
    # Calcul des prédictions pour affichage final
    train_preds_qru = torch.stack([circuit_qru(params_qru, x) for x in X_train]).detach().numpy()
    test_preds_qru = torch.stack([circuit_qru(params_qru, x) for x in X_test]).detach().numpy()
    train_preds_qru_qrb = torch.stack([circuit_qru_qrb(params_qru_qrb, x) for x in X_train]).detach().numpy()
    test_preds_qru_qrb = torch.stack([circuit_qru_qrb(params_qru_qrb, x) for x in X_test]).detach().numpy()
    
    train_preds_qru_list.append(train_preds_qru)
    test_preds_qru_list.append(test_preds_qru)
    train_preds_qru_qrb_list.append(train_preds_qru_qrb)
    test_preds_qru_qrb_list.append(test_preds_qru_qrb)

    df_train_pred_qru = pd.DataFrame({
    "Y_train": Y_train.detach().numpy(),
    "train_preds_pqc": train_preds_qru_list[-1],    
    })
    df_test_pred_qru = pd.DataFrame({    
    "Y_test": Y_test.detach().numpy(),
    "test_preds_pqc": test_preds_qru_list[-1]
    })

    df_train_pred_qru_qrb = pd.DataFrame({
    "Y_train": Y_train.detach().numpy(),
    "train_preds_qru": train_preds_qru_qrb_list[-1].squeeze(),    
    })
    df_test_pred_qru_qrb = pd.DataFrame({ 
    "Y_test": Y_test.detach().numpy(),   
    "test_preds_qru": test_preds_qru_qrb_list[-1].squeeze(),
    })
    
    log_line = f"{epoch}, {train_loss_qru:.6f}, {test_loss_qru:.6f}, {train_acc_qru:.6f}, {test_acc_qru:.6f}, {grad_norm_qru:.6f}, " \
               f"{train_loss_qru_qrb:.6f}, {test_loss_qru_qrb:.6f}, {train_acc_qru_qrb:.6f}, {test_acc_qru_qrb:.6f}, {grad_norm_qru_qrb:.6f}\n"
    log_file.write(log_line)
    log_file.flush()
    
    print(f"Epoch {epoch}: QRU -> Train Loss: {train_loss_qru:.4f}, Test Loss: {test_loss_qru:.4f}, "
          f"Train Acc: {train_acc_qru:.4f}, Test Acc: {test_acc_qru:.4f}, Grad Norm: {grad_norm_qru:.4f} || "
          f"QRU_QRB -> Train Loss: {train_loss_qru_qrb:.4f}, Test Loss: {test_loss_qru_qrb:.4f}, "
          f"Train Acc: {train_acc_qru_qrb:.4f}, Test Acc: {test_acc_qru_qrb:.4f}, Grad Norm: {grad_norm_qru_qrb:.4f}")
log_file.close()

df_train_pred_qru_qrb.to_csv("River_QRU_vs_QRB_14.03/PQC_train_predictions.csv", index=False)
df_test_pred_qru_qrb.to_csv("River_QRU_vs_QRB_14.03/PQC_test_predictions.csv", index=False)
df_train_pred_qru.to_csv("River_QRU_vs_QRB_14.03/QRU_train_predictions.csv", index=False)
df_test_pred_qru.to_csv("River_QRU_vs_QRB_14.03/QRU_test_predictions.csv", index=False)

epochs = range(num_epochs)

# ===== Sauvegarde des figures =====
# Comparaison des Loss
plt.figure()
plt.plot(epochs, train_loss_qru_list, label="QRU Train Loss")
plt.plot(epochs, test_loss_qru_list, label="QRU Test Loss")
plt.plot(epochs, train_loss_qru_qrb_list, label="QRU_QRB Train Loss")
plt.plot(epochs, test_loss_qru_qrb_list, label="QRU_QRB Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs Comparison")
plt.legend()
plt.savefig(os.path.join(results_dir, "loss_comparison.png"), dpi=300)
plt.close()

# Comparaison des Accuracy
plt.figure()
plt.plot(epochs, train_acc_qru_list, label="QRU Train Accuracy")
plt.plot(epochs, test_acc_qru_list, label="QRU Test Accuracy")
plt.plot(epochs, train_acc_qru_qrb_list, label="QRU_QRB Train Accuracy")
plt.plot(epochs, test_acc_qru_qrb_list, label="QRU_QRB Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs Comparison")
plt.legend()
plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"), dpi=300)
plt.close()

# Comparaison des normes des gradients (optionnel)
plt.figure()
plt.plot(epochs, grad_norm_qru_list, label="QRU Grad Norm")
plt.plot(epochs, grad_norm_qru_qrb_list, label="QRU_QRB Grad Norm")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm over Epochs Comparison")
plt.legend()
plt.savefig(os.path.join(results_dir, "grad_norm_comparison.png"), dpi=300)
plt.close()

# Prédictions vs Valeurs réelles pour l'ensemble d'entraînement
plt.figure()
plt.plot(train_preds_qru_list[-1], 'r-', label="QRU Predictions")
plt.plot(train_preds_qru_qrb_list[-1], 'g-', label="QRU_QRB Predictions")
plt.plot(Y_train.detach().numpy(), 'b--', label="Actual Values")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("Train : Predictions vs Actual Values Comparison")
plt.legend()
plt.savefig(os.path.join(results_dir, "train_predictions_comparison.png"), dpi=300)
plt.close()

# Prédictions vs Valeurs réelles pour l'ensemble de test
plt.figure()
plt.plot(test_preds_qru_list[-1], 'r-', label="QRU Predictions")
plt.plot(test_preds_qru_qrb_list[-1], 'g-', label="QRU_QRB Predictions")
plt.plot(Y_test.detach().numpy(), 'b--', label="Valeurs réelles")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.title("Test : Predictions vs Actual Values Comparison")
plt.legend()
plt.savefig(os.path.join(results_dir, "test_predictions_comparison.png"), dpi=300)
plt.close()

print("Experiments completed", results_dir)
