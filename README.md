# Quantum Reupload Units (QRU)
[![Paper](https://img.shields.io/badge/Paper-QCE25-blue)](link_to_paper_or_arxiv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Pennylane](https://img.shields.io/badge/Pennylane-compatible-brightgreen)](https://pennylane.ai/)

A hardware-efficient, single-qubit quantum architecture tailored for time series forecasting. This repository provides code, benchmarks, and analysis for the models described in the QCE25 paper: **"Quantum Reupload Units: A Scalable and Expressive Approach for Time Series Learning"**.


## 📄 Paper
> [Quantum Reupload Units: A Scalable and Expressive Approach for Time Series Learning](link_to_pdf_or_arxiv_if_applicable)

## ✨ Highlights
- Single-qubit quantum architecture with re-uploaded inputs
- Demonstrates superior expressivity and convergence over PQC, VQC, and RNNs
- Benchmarked on both synthetic and real-world time series datasets
- Fourier-based expressivity and absorption witness analysis

## 🧠 Architectures
- QRU (Quantum Reupload Unit)
- PQC (Parameterized Quantum Circuit)
- VQC (Variational Quantum Circuit)
- QRU–QRB–Local (with shared ancilla)
- QRU–QRB–Global (with independent ancilla)

📊 Datasets

Dataset	Source
Mackey-Glass	Synthetic chaotic time series
Sinusoidal Wave	Simple periodic series
River Level	TAIAO Project (Real-world dataset)

📌 Features

✅ Supports multiple quantum architectures: PQC, VQC, QRU, QRU-QRB (local/global)

✅ Fourier expressivity analysis via amplitude spectrum

✅ Absorption witness & KL divergence computation

✅ Realistic comparison with RNN using parameter-matched setup

✅ Easy extensibility for custom datasets

## 📖 Citation

```bibtex
@inproceedings{casse2025qru,
  title={Quantum Reupload Units: A Scalable and Expressive Approach for Time Series Learning},
  author={Cassé, Léa and Ponnambalam, Sabarikirishwaran and Pfahringer, Bernhard and Bifet, Albert},
  booktitle={IEEE Quantum Week (QCE25)},
  year={2025}
}
```

🧠 Authors

Léa Cassé – University of Waikato & École Polytechnique

Sabarikirishwaran Ponnambalam – Griffith University

Bernhard Pfahringer – University of Waikato

Albert Bifet – University of Waikato & Télécom Paris

🏷️ Tags & Topics

#QuantumML #TimeSeriesForecasting #Pennylane #QML #NISQ #QiskitCompatible #QuantumExpressivity #FourierAnalysis #GradientFlow #QuantumCircuitDesign #AbsorptionWitness

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.
