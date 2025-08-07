import pennylane as qml
from pennylane import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# ==============================================================================
# 1. CONFIGURATION ET INITIALISATION
# ==============================================================================
print("Initialisation du script...")

# Fixer les graines pour la reproductibilité
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# --- Paramètres globaux de l'expérience ---
# Nombre d'échantillons de paramètres pour calculer la variance
N_SAMPLES_VARIANCE = 100
# Profondeur maximale du QRU à tester
MAX_QRU_DEPTH = 15
# Longueur de la séquence d'entrée (nombre de features pour x)
SEQ_LENGTH = 3
# Nombre de qubits pour les circuits
NB_QUBITS = 2 # Augmenté à 2 pour des architectures plus intéressantes

# Création du device PennyLane pour les simulations
dev = qml.device("default.qubit", wires=NB_QUBITS, shots=None)

# Création du dossier pour les données et les résultats si nécessaire
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('results'):
    os.makedirs('results')

# ==============================================================================
# 2. FONCTIONS DE PRÉPARATION DES DONNÉES
# ==============================================================================

def generate_sine_data(n_points=100, seq_length=3):
    """Génère une série temporelle sinusoïdale simple."""
    time_steps = np.linspace(0, 4 * np.pi, n_points + seq_length)
    data = np.sin(time_steps)
    # Normalisation entre -1 et 1
    return 2 * (data - data.min()) / (data.max() - data.min()) - 1

def generate_mackey_glass(n_points=2000, tau=17, dt=1.0, beta=0.2, gamma=0.1, n=10, initial_value=1.2):
    """Génère la série temporelle de Mackey-Glass."""
    hist_len = int(np.ceil(tau / dt))
    total_len = n_points + hist_len
    s = np.zeros(total_len)
    s[:hist_len] = initial_value
    for t in range(hist_len, total_len):
        s[t] = s[t-1] + dt * (beta * s[t-hist_len] / (1.0 + s[t-hist_len]**n) - gamma * s[t-1])
    data = s[hist_len:]
    # Normalisation entre -1 et 1
    return 2 * (data - data.min()) / (data.max() - data.min()) - 1

def preprocess_river_data(file_path='data/river.csv'):
    """Charge et normalise les données de niveau de rivière."""
    try:
        df = pd.read_csv(file_path, skiprows=2, header=None)
        df.columns = ['date', 'wlvalue', 'fvalue']
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # --- DÉBUT DU CORRECTIF ---
        # Forcer la conversion de la colonne 'wlvalue' en type numérique.
        # errors='coerce' transformera toute valeur non convertible en NaN (Not a Number).
        df['wlvalue'] = pd.to_numeric(df['wlvalue'], errors='coerce')
        # --- FIN DU CORRECTIF ---

        # Supprimer les lignes où la date ou la valeur sont maintenant manquantes (NaN)
        df = df.dropna(subset=['date', 'wlvalue'])
        
        # Vérifier s'il reste des données après le nettoyage
        if df.empty:
            print(f"ATTENTION: Aucune donnée valide n'a pu être lue depuis {file_path}. La colonne 'wlvalue' est peut-être vide ou ne contient que des valeurs non numériques.")
            return None

        df.set_index('date', inplace=True)
        data = df['wlvalue'].values

        # Cette ligne fonctionnera maintenant car 'data' est un tableau de nombres
        return 2 * (data - data.min()) / (data.max() - data.min()) - 1

    except FileNotFoundError:
        print(f"ERREUR: Le fichier {file_path} est introuvable.")
        print("Veuillez créer un fichier CSV à cet emplacement ou modifier le chemin.")
        return None

# ==============================================================================
# 3. DÉFINITION DES ARCHITECTURES QUANTIQUES
# ==============================================================================

def encoding_layer(x):
    """Couche d'encodage des données x."""
    for q in range(NB_QUBITS):
        qml.RX(x[0] * np.pi, wires=q)
        qml.RY(x[1] * np.pi, wires=q)
        qml.RZ(x[2] * np.pi, wires=q)

def variational_layer(params, architecture_name="basic"):
    """
    Couche variationnelle paramétrée.
    - basic: Simples rotations sur chaque qubit.
    - entangled: Rotations suivies de portes CNOT en chaîne.
    - strongly_entangling: Template de PennyLane pour un fort enchevêtrement.
    """
    if architecture_name == "basic":
        for q in range(NB_QUBITS):
            qml.RX(params[3*q], wires=q)
            qml.RY(params[3*q + 1], wires=q)
            qml.RZ(params[3*q + 2], wires=q)
    elif architecture_name == "entangled":
        for q in range(NB_QUBITS):
            qml.RX(params[3*q], wires=q)
            qml.RY(params[3*q + 1], wires=q)
            qml.RZ(params[3*q + 2], wires=q)
        for q in range(NB_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])
    elif architecture_name == "strongly_entangling":
        # Cette couche a 3*NB_QUBITS paramètres par défaut
        qml.StronglyEntanglingLayers(params.reshape(1, NB_QUBITS, 3), wires=range(NB_QUBITS))
    else:
        raise ValueError("Nom d'architecture non reconnu.")

def get_params_per_layer(architecture_name):
    """Retourne le nombre de paramètres pour une couche variationnelle."""
    # Toutes nos architectures utilisent 3 rotations par qubit
    return 3 * NB_QUBITS

# --- Définition des QNodes ---

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def qru_circuit(params, x, qru_depth, architecture_name):
    """Circuit du Quantum Re-uploading Unit (QRU)."""
    params_per_layer = get_params_per_layer(architecture_name)
    # Nous n'avons pas de paramètres d'encodage dédiés, donc tout est variationnel
    
    for d in range(qru_depth):
        encoding_layer(x)
        layer_params = params[d * params_per_layer : (d + 1) * params_per_layer]
        variational_layer(layer_params, architecture_name)
        
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def pqc_circuit(params, x, pqc_depth, architecture_name):
    """Circuit PQC de base avec un seul encodage."""
    params_per_layer = get_params_per_layer(architecture_name)
    
    encoding_layer(x) # Encodage unique au début
    
    for d in range(pqc_depth):
        layer_params = params[d * params_per_layer : (d + 1) * params_per_layer]
        variational_layer(layer_params, architecture_name)
        
    return qml.expval(qml.PauliZ(0))


# ==============================================================================
# 4. FONCTION PRINCIPALE DE CALCUL DE AW_num
# ==============================================================================

def compute_aw_metric(x_input, qru_depth, architecture_name):
    """
    Calcule la métrique AW_num pour une configuration donnée.
    AW_num = Var_theta(||grad_QRU||) - Var_theta(||grad_PQC||)
    """
    # 1. Déterminer le nombre de paramètres et les profondeurs
    params_per_variational_layer = get_params_per_layer(architecture_name)
    
    # Pour le QRU, les paramètres sont répétés à chaque couche de re-uploading
    total_params_qru = qru_depth * params_per_variational_layer
    
    # Pour le PQC, on ajuste sa profondeur pour avoir le même nombre de paramètres
    # Comme il n'y a pas de paramètres d'encodage, le calcul est direct.
    pqc_depth = qru_depth # Chaque couche (encodage+var) du QRU correspond à une couche var du PQC
    total_params_pqc = pqc_depth * params_per_variational_layer
    
    # Assertion pour la sécurité, mais ils devraient être égaux par construction
    assert total_params_qru == total_params_pqc, "Le nombre de paramètres doit être identique !"
    
    # Liste pour stocker la norme des gradients pour chaque échantillon de theta
    qru_grad_norms = []
    pqc_grad_norms = []

    for _ in range(N_SAMPLES_VARIANCE):
        # 2. Échantillonner des paramètres aléatoires dans [-pi, pi]
        # On crée un seul vecteur de paramètres que les deux circuits utiliseront
        params_tensor = torch.tensor(
            np.random.uniform(-np.pi, np.pi, size=total_params_qru),
            dtype=torch.float64,
            requires_grad=True
        )

        # 3. Calculer la norme du gradient pour le QRU
        qru_out = qru_circuit(params_tensor, x_input, qru_depth, architecture_name)
        qru_out.backward()
        qru_grad = params_tensor.grad.clone() # Cloner pour éviter les interférences
        qru_grad_norm = torch.norm(qru_grad)
        qru_grad_norms.append(qru_grad_norm.item())
        params_tensor.grad.zero_() # Remettre le gradient à zéro

        # 4. Calculer la norme du gradient pour le PQC
        pqc_out = pqc_circuit(params_tensor, x_input, pqc_depth, architecture_name)
        pqc_out.backward()
        pqc_grad = params_tensor.grad.clone()
        pqc_grad_norm = torch.norm(pqc_grad)
        pqc_grad_norms.append(pqc_grad_norm.item())
        # pas besoin de zero_() car c'est la fin de la boucle

    # 5. Calculer la variance des normes de gradient
    var_qru = np.var(qru_grad_norms)
    var_pqc = np.var(pqc_grad_norms)
    
    # 6. Calculer AW_num
    aw_num = var_qru - var_pqc
    
    return aw_num, var_qru, var_pqc

# ==============================================================================
# 5. BOUCLE PRINCIPALE D'EXÉCUTION
# ==============================================================================

if __name__ == "__main__":
    
    print("Préparation des datasets...")
    # Charger les données et prendre un seul point d'entrée x pour les calculs
    # (AW est calculé pour un x fixe)
    datasets = {
        "Sinusoïde": generate_sine_data()[:SEQ_LENGTH],
        "Mackey-Glass": generate_mackey_glass()[:SEQ_LENGTH],
        "Niveau Rivière": preprocess_river_data()
    }
    
    # Filtrer les datasets qui n'ont pas pu être chargés
    datasets = {name: data[:SEQ_LENGTH] for name, data in datasets.items() if data is not None}
    
    architectures = ["basic", "entangled", "strongly_entangling"]
    depths = range(1, MAX_QRU_DEPTH + 1)
    
    results = {}
    
    start_time = time.time()
    
    print("\nDébut du calcul de l'Absorption Witness (AW)...")
    print(f"Configurations à tester: {len(datasets)} datasets x {len(architectures)} architectures x {len(depths)} profondeurs.")
    
    for dataset_name, x_input in datasets.items():
        print(f"\n--- TRAITEMENT DU DATASET : {dataset_name} ---")
        results[dataset_name] = {}
        x_input_tensor = torch.tensor(x_input, dtype=torch.float64)
        
        for arch_name in architectures:
            print(f"  Architecture : {arch_name}")
            results[dataset_name][arch_name] = []
            
            for depth in depths:
                aw_value, _, _ = compute_aw_metric(x_input_tensor, depth, arch_name)
                results[dataset_name][arch_name].append(aw_value)
                print(f"    Profondeur QRU = {depth:2d} -> AW_num = {aw_value:.6f}")

    end_time = time.time()
    print(f"\nCalculs terminés en {end_time - start_time:.2f} secondes.")

    # ==============================================================================
    # 6. VISUALISATION DES RÉSULTATS
    # ==============================================================================
    
    print("Génération des graphiques de résultats...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    num_datasets = len(datasets)
    fig, axes = plt.subplots(num_datasets, 1, figsize=(12, 7 * num_datasets), sharex=True)
    if num_datasets == 1:
        axes = [axes] # Rendre `axes` itérable s'il n'y a qu'un seul subplot

    for i, (dataset_name, arch_results) in enumerate(results.items()):
        ax = axes[i]
        for arch_name, aw_values in arch_results.items():
            ax.plot(depths, aw_values, marker='o', linestyle='-', label=f"Arch: {arch_name}")
        
        ax.axhline(0, color='black', linewidth=1.5, linestyle='--', label="AW = 0 (Absorption idéale)")
        ax.set_title(f"Absorption Witness (AW_num) vs. Profondeur du QRU\nDataset: {dataset_name}", fontsize=14)
        ax.set_ylabel("AW_num = Var(||∇QRU||) - Var(||∇PQC||)", fontsize=12)
        ax.legend(title="Architecture")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    axes[-1].set_xlabel("Profondeur du QRU (Nombre de blocs de ré-encodage)", fontsize=12)
    plt.tight_layout()
    
    plot_filename = 'results/absorption_witness_analysis.png'
    plt.savefig(plot_filename)
    print(f"Graphique sauvegardé sous : {plot_filename}")
    
    plt.show()