import os
import torch
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Pour la reproductibilité
np.random.seed(42)
torch.manual_seed(42)

# ===== Paramètres communs =====
n_input = 3             # Taille de la fenêtre
total_points = 500      # Nombre total de points simulés pour Mackey Glass
train_ratio = 0.7
num_epochs = 300
learning_rate = 0.01
delta_huber = 0.1       # Paramètre de la Huber loss
accuracy_threshold = 0.1

# Profondeurs à tester
depths_to_test = list(range(1, 11))  # [1, 2, ..., 10]

# Création du répertoire pour enregistrer les résultats
results_dir = "MG_QRU_depth_comparison_13.03"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ===== Génération de la série de Mackey Glass =====
def generate_mackey_glass(total_steps, tau=17, n=10, beta=0.2, gamma=0.1, dt=1):
    """
    Simulation de la série de Mackey Glass par méthode d'Euler.
    Conditions initiales (pour t < tau) fixées à 1.2.
    """
    history = np.zeros(total_steps + tau)
    history[:tau] = 1.2
    for t in range(tau, total_steps + tau):
        history[t] = history[t-1] + dt * (beta * history[t-tau] / (1 + history[t-tau]**n) - gamma * history[t-1])
    return history[tau:]

mackey_series = generate_mackey_glass(total_points)

# ===== Normalisation de la série entre -1 et 1 =====
min_val = np.min(mackey_series)
max_val = np.max(mackey_series)
mackey_series_norm = 2 * (mackey_series - min_val) / (max_val - min_val) - 1

# ===== Création des fenêtres glissantes =====
def create_windows(series, window_size):
    X = []
    Y = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        Y.append(series[i+window_size])
    return np.array(X), np.array(Y)

X_all, Y_all = create_windows(mackey_series_norm, n_input)

# Conversion en torch.tensor
X_all = torch.tensor(X_all, dtype=torch.float64)
Y_all = torch.tensor(Y_all, dtype=torch.float64)

# Séparation en ensembles d'entraînement et de test
split_idx = int(len(X_all) * train_ratio)
X_train_all, Y_train_all = X_all[:split_idx], Y_all[:split_idx]
X_test_all, Y_test_all = X_all[split_idx:], Y_all[split_idx:]

# Dictionnaire pour stocker les résultats par profondeur
results = {}

# ===== Boucle sur les différentes profondeurs =====
for current_depth in depths_to_test:
    print(f"\nEntraînement pour depth = {current_depth}")
    
    # Définition du device et du circuit quantique pour la profondeur courante
    dev = qml.device("default.qubit", wires=1)
    
    # Circuit quantique avec re-uploading : la profondeur est capturée par la variable current_depth
    def QRU_local(params, x):
        for i in range(current_depth):
            for j in range(len(x)):
                qml.RX(params[i, 3 * j], wires=0)
                qml.RY(params[i, 3 * j + 1] * x[j], wires=0)
                qml.RZ(params[i, 3 * j + 2], wires=0)
        return qml.expval(qml.PauliZ(0))
    
    @qml.qnode(dev, interface="torch")
    def circuit_local(params, x):
        return QRU_local(params, x)
    
    # ===== Définition de la Huber loss et des métriques =====
    def huber_loss(y_true, y_pred, delta=delta_huber):
        error = y_true - y_pred
        is_small_error = torch.abs(error) <= delta
        squared_loss = 0.5 * error**2
        linear_loss = delta * (torch.abs(error) - 0.5 * delta)
        return torch.where(is_small_error, squared_loss, linear_loss)
    
    def loss_fn(params, X, Y, delta=delta_huber):
        preds = torch.stack([circuit_local(params, x) for x in X])
        return torch.mean(huber_loss(Y, preds, delta))
    
    def accuracy_fn(params, X, Y, threshold=accuracy_threshold):
        preds = torch.stack([circuit_local(params, x) for x in X])
        correct = torch.sum(torch.abs(preds - Y) < threshold)
        return correct.item() / len(Y)
    
    # ===== Initialisation des paramètres pour la profondeur courante =====
    # Le circuit requiert 3 paramètres par entrée, d'où la taille (current_depth, 3*n_input)
    params = torch.tensor(np.full((current_depth, 3 * n_input), 0.5),
                          requires_grad=True, dtype=torch.float64)
    
    # Optimiseur Adam
    optimizer = torch.optim.Adam([params], lr=learning_rate)
    
    # Listes pour sauvegarder les métriques pour la profondeur courante
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    grad_norm_list = []
    train_preds_last = None  # Prédictions à la dernière époque (train)
    test_preds_last = None   # Prédictions à la dernière époque (test)
    
    # ===== Boucle d'entraînement =====
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_fn(params, X_train_all, Y_train_all)
        loss.backward()
        grad_norm_val = params.grad.norm().item() if params.grad is not None else 0
        optimizer.step()
        
        # Calcul des métriques après mise à jour
        train_loss = loss_fn(params, X_train_all, Y_train_all).item()
        test_loss = loss_fn(params, X_test_all, Y_test_all).item()
        train_acc = accuracy_fn(params, X_train_all, Y_train_all)
        test_acc = accuracy_fn(params, X_test_all, Y_test_all)
        train_preds = torch.stack([circuit_local(params, x) for x in X_train_all]).detach().numpy()
        test_preds = torch.stack([circuit_local(params, x) for x in X_test_all]).detach().numpy()
        
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        grad_norm_list.append(grad_norm_val)
        
        # Sauvegarde des prédictions de la dernière époque
        if epoch == num_epochs - 1:
            train_preds_last = train_preds
            test_preds_last = test_preds
        
        if epoch % 50 == 0:
            print(f"Depth {current_depth}, Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Grad Norm: {grad_norm_val:.4f}")
    
    # Stockage des résultats pour la profondeur courante
    results[current_depth] = {
        "train_loss": train_loss_list,
        "test_loss": test_loss_list,
        "train_acc": train_acc_list,
        "test_acc": test_acc_list,
        "grad_norm": grad_norm_list,
        "train_preds": train_preds_last,
        "test_preds": test_preds_last,
    }
    
    # Sauvegarde locale des métriques pour la profondeur courante
    with open(os.path.join(results_dir, f"metrics_depth_{current_depth}.txt"), "w") as f:
        f.write("epoch, train_loss, test_loss, train_accuracy, test_accuracy, grad_norm\n")
        for epoch in range(num_epochs):
            f.write(f"{epoch}, {train_loss_list[epoch]:.6f}, {test_loss_list[epoch]:.6f}, "
                    f"{train_acc_list[epoch]:.6f}, {test_acc_list[epoch]:.6f}, {grad_norm_list[epoch]:.6f}\n")

# ===== Tracé des figures comparatives =====
epochs = range(num_epochs)
# On utilise une palette de 10 couleurs distinctes
colors = plt.cm.tab10(np.linspace(0, 1, len(depths_to_test)))

# --- Figure 1 : Loss (Train et Test dans deux sous-graphes) ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for d in depths_to_test:
    plt.plot(epochs, results[d]["train_loss"], label=f"Depth {d}")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Train Loss over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
for d in depths_to_test:
    plt.plot(epochs, results[d]["test_loss"], label=f"Depth {d}")
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.title("Test Loss over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "loss_comparison.png"), dpi=300)
plt.close()

# --- Figure 2 : Accuracy (Train et Test) ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for d in depths_to_test:
    plt.plot(epochs, results[d]["train_acc"], label=f"Depth {d}")
plt.xlabel("Epoch")
plt.ylabel("Train Accuracy")
plt.title("Train Accuracy over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
for d in depths_to_test:
    plt.plot(epochs, results[d]["test_acc"], label=f"Depth {d}")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"), dpi=300)
plt.close()

# --- Figure 3 : Gradient Norm over Epochs ---
plt.figure()
for d in depths_to_test:
    plt.plot(epochs, results[d]["grad_norm"], label=f"Depth {d}")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm over Epochs")
plt.legend()
plt.savefig(os.path.join(results_dir, "grad_norm_comparison.png"), dpi=300)
plt.close()

# --- Figure 4 : Train Predictions vs Valeurs Réelles ---
plt.figure()
plt.plot(range(len(Y_train_all)), Y_train_all.detach().numpy(), 'k--', label="Valeurs Réelles")
for d in depths_to_test:
    plt.plot(range(len(Y_train_all)), results[d]["train_preds"], label=f"Depth {d}")
plt.xlabel("Index d'échantillon")
plt.ylabel("Valeur")
plt.title("Train : Prédictions vs Valeurs Réelles")
plt.legend()
plt.savefig(os.path.join(results_dir, "train_predictions_comparison.png"), dpi=300)
plt.close()

# --- Figure 5 : Test Predictions vs Valeurs Réelles ---
plt.figure()
plt.plot(range(len(Y_test_all)), Y_test_all.detach().numpy(), 'k--', label="Valeurs Réelles")
for d in depths_to_test:
    plt.plot(range(len(Y_test_all)), results[d]["test_preds"], label=f"Depth {d}")
plt.xlabel("Index d'échantillon")
plt.ylabel("Valeur")
plt.title("Test : Prédictions vs Valeurs Réelles")
plt.legend()
plt.savefig(os.path.join(results_dir, "test_predictions_comparison.png"), dpi=300)
plt.close()

print("Entraînement terminé pour toutes les profondeurs. Les métriques et figures ont été sauvegardées dans le répertoire", results_dir)
