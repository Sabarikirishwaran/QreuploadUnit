import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

# ===== General Parameters =====
depth = 6
alpha = 0.5
n_input = 3 
results_dir = "plot_summary_all_models"

# Create folders if they don’t exist
os.makedirs(os.path.join(results_dir, "circuit_summaries"), exist_ok=True)
os.makedirs(os.path.join(results_dir, "circuit_visuals"), exist_ok=True)

# ===== Quantum Circuit Definitions =====

# 1. QRU
dev_qru = qml.device("default.qubit", wires=1)
@qml.qnode(dev_qru)
def circuit_qru(params, x):
    for i in range(depth):
        for j in range(len(x)):
            qml.RX(params[i][3*j], wires=0)
            qml.RY(params[i][3*j + 1] * x[j], wires=0)
            qml.RZ(params[i][3*j + 2], wires=0)
    return qml.expval(qml.PauliZ(0))

# 2. VQC
dev_vqc = qml.device("default.qubit", wires=n_input)
@qml.qnode(dev_vqc)
def circuit_vqc(params, x):
    for j in range(len(x)):
        qml.RY(x[j], wires=j)
    for i in range(depth):
        for j in range(len(x)):
            qml.RX(params[i][j], wires=j)
        entangler_params = params[i][len(x):].reshape(1, len(x))
        qml.templates.BasicEntanglerLayers(entangler_params, wires=range(len(x)))
    return qml.expval(qml.PauliZ(0))

# 3. PQC
dev_pqc = qml.device("default.qubit", wires=1)
@qml.qnode(dev_pqc)
def circuit_pqc(params, x):
    for j in range(len(x)):
        qml.RY(x[j], wires=0)
    for i in range(depth):
        for j in range(len(x)):
            qml.RX(params[i][2*j], wires=0)
            qml.RY(params[i][2*j + 1], wires=0)
            qml.RZ(params[i][3*j + 2], wires=0)
    return qml.expval(qml.PauliZ(0))

# 4. QRU_QRB_Local
dev_qru_qrb1 = qml.device("default.qubit", wires=2)
@qml.qnode(dev_qru_qrb1)
def circuit_qru_qrb1(params, x, alpha=0.5):
    qml.Hadamard(wires=1)
    for i in range(depth):
        for j in range(len(x)):
            qml.RY(params[i][3*j] * x[j], wires=0)
        qml.CRX(alpha * params[i][3*len(x) + 1], wires=[1, 0])
        for j in range(len(x)):
            qml.RX(params[i][3*j + 1], wires=0)
            qml.RZ(params[i][3*j + 2], wires=0)
    qml.Hadamard(wires=1)
    proj = (qml.Identity(wires=1) + qml.PauliZ(wires=1)) / 2
    return qml.expval(qml.PauliZ(0) @ proj)

# 5. QRU_QRB_Global
dev_qru_qrbL = qml.device("default.qubit", wires=depth+1)
@qml.qnode(dev_qru_qrbL)
def circuit_qru_qrbL(params, x, alpha=0.5):
    for i in range(depth):
        qml.Hadamard(wires=i+1)
        for j in range(len(x)):
            qml.RY(params[i][3*j] * x[j], wires=0)
        qml.CRX(alpha * params[i][3*len(x) + 1], wires=[i+1, 0])
        for j in range(len(x)):
            qml.RX(params[i][3*j + 1], wires=0)
            qml.RZ(params[i][3*j + 2], wires=0)
        qml.Hadamard(wires=i+1)
    obs = qml.PauliZ(0)
    for i in range(1, depth+1):
        obs = obs @ ((qml.Identity(wires=i) + qml.PauliZ(wires=i)) / 2)
    return qml.expval(obs)

# 6. QRU with entanglement
dev_qru_entanglement = qml.device("default.qubit", wires=2)
@qml.qnode(dev_qru_entanglement)
def circuit_qru_entanglement(params, x, alpha=0.5):
    for i in range(depth):
        qml.RX(params[i][0], wires=1)
        for j in range(len(x)):
            qml.RX(params[i][3*j + 1], wires=0)
            qml.RY(params[i][3*j + 2] * x[j], wires=0)
            qml.RZ(params[i][3*j + 3], wires=0)
        qml.CRX(alpha * params[i][3*len(x) + 1], wires=[1, 0])
    return qml.expval(qml.PauliZ(0))

# ===== Parameter Initialization =====
params_qru              = np.full((depth, 3 * n_input), 0.5)
params_vqc              = np.full((depth, 2 * n_input), 0.5)
params_pqc              = np.full((depth, 3 * n_input), 0.5)
params_qru_qrb1         = np.full((depth, 3 * n_input + 2), 0.5)
params_qru_qrbL         = np.full((depth, 3 * n_input + 2), 0.5)
params_qru_entanglement = np.full((depth, 3 * n_input + 2), 0.5)

dummy_input = [1.0] * n_input

# ===== Circuit Summary and Diagram Functions =====

def save_circuit_summaries():
    """Save textual summaries of each circuit in .txt files."""
    summary_dir = os.path.join(results_dir, "circuit_summaries")

    circuits = {
        "QRU": (circuit_qru, params_qru, dummy_input),
        "VQC": (circuit_vqc, params_vqc, dummy_input),
        "PQC": (circuit_pqc, params_pqc, dummy_input),
        "QRU_QRB_1": (circuit_qru_qrb1, params_qru_qrb1, dummy_input),
        "QRU_QRB_L": (circuit_qru_qrbL, params_qru_qrbL, dummy_input),
        "QRU_entanglement": (circuit_qru_entanglement, params_qru_entanglement, dummy_input)
    }
    for name, (qnode, dummy_params, dummy_input_args) in circuits.items():
        tape_fn = qml.tape.make_qscript(qnode.func)
        tape = tape_fn(dummy_params, dummy_input_args)
        n_gates = len(tape.operations)
        n_qubits = qnode.device._wires
        param_count = np.array(dummy_params).size
        summary_text = f"Circuit summary for {name}:\n"
        summary_text += f"Depth: {depth}\n"
        summary_text += f"Total trainable parameters: {param_count}\n"
        summary_text += f"Number of qubits: {n_qubits}\n"
        summary_text += f"Number of quantum gates: {n_gates}\n"
        
        file_path = os.path.join(summary_dir, f"{name}_summary.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"{name} summary saved to {file_path}")

def print_model_diagram(qnode, dummy_params, dummy_input, name, *args):
    """Generate and save the circuit diagram as PNG."""
    draw_func = qml.draw_mpl(qnode, decimals=2, style="pennylane")
    fig, ax = draw_func(dummy_params, dummy_input, *args)
    fig.suptitle("Quantum Circuit Diagram: " + name)
    save_path = os.path.join(results_dir, "circuit_visuals", f"{name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"{name} diagram saved to {save_path}")

def save_circuit_diagrams():
    """Save all circuit diagrams."""
    print_model_diagram(circuit_qru, params_qru, dummy_input, "QRU")
    print_model_diagram(circuit_vqc, params_vqc, dummy_input, "VQC")
    print_model_diagram(circuit_pqc, params_pqc, dummy_input, "PQC")
    print_model_diagram(circuit_qru_qrb1, params_qru_qrb1, dummy_input, "QRU_QRB_1", alpha)
    print_model_diagram(circuit_qru_qrbL, params_qru_qrbL, dummy_input, "QRU_QRB_L", alpha)
    print_model_diagram(circuit_qru_entanglement, params_qru_entanglement, dummy_input, "QRU_entanglement", alpha)

# ===== Main =====
def main():
    parser = argparse.ArgumentParser(description="Save summaries and diagrams of quantum circuits.")
    parser.add_argument("--summary", action="store_true", help="Save circuit summary in text format.")
    parser.add_argument("--diagrams", action="store_true", help="Save circuit diagrams as PNG.")
    args = parser.parse_args()

    # If no argument is provided, save everything
    if not (args.summary or args.diagrams):
        args.summary = args.diagrams = True

    if args.summary:
        print("Saving circuit summaries...")
        save_circuit_summaries()

    if args.diagrams:
        print("Saving circuit diagrams...")
        save_circuit_diagrams()

if __name__ == "__main__":
    main()
