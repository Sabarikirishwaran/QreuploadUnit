# Quantum Reupload Units (QRU)

A hardware-efficient, single-qubit quantum architecture tailored for time series forecasting. This repository provides code, benchmarks, and analysis for the models described in the QCE25 paper: **"Quantum Reupload Units: A Scalable and Expressive Approach for Time Series Learning"**.

## 📄 Paper
> [Quantum Reupload Units: A Scalable and Expressive Approach for Time Series Learning](link_to_pdf_or_arxiv_if_applicable)

## ✨ Highlights
- Single-qubit quantum architecture with re-uploaded inputs
- Demonstrates superior expressivity and convergence over PQC, VQC, and RNNs
- Benchmarked on both synthetic and real-world time series datasets
- Fourier-based expressivity and absorption witness analysis

## 🧠 Key Architectures
- QRU (Quantum Reupload Unit)
- PQC (Parameterized Quantum Circuit)
- VQC (Variational Quantum Circuit)
- QRU–QRB–Local (with shared ancilla)
- QRU–QRB–Global (with independent ancilla)

## 📁 Project Structure

```bash
.
├── qru_model.py                  # QRU architecture implementation
├── pqc_model.py                 # PQC baseline model
├── vqc_model.py                 # VQC baseline model
├── qru_qrb_local.py            # QRU with QRB (shared ancilla)
├── qru_qrb_global.py           # QRU with QRB (global ancilla)
├── train.py                     # Training loop for all architectures
├── data_utils.py                # Data generation: synthetic & real
├── evaluate.py                  # Evaluation metrics and loss plots
├── fft_analysis.py              # Fourier-based expressivity analysis
├── absorption_witness.py        # Absorption witness implementation
├── plot_utils.py                # Common plotting utilities
├── requirements.txt             # List of dependencies
└── README.md
