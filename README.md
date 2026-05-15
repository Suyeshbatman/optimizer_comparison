---
title: Optimizer Benchmark
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.45.1"
app_file: app/streamlit_app.py
pinned: false
---

# Optimizer Benchmark Suite

A Streamlit application for training, comparing, and evaluating PyTorch optimizers across multiple datasets and model architectures. Built for understanding how optimizer choice affects convergence, accuracy, memory usage, and training speed.

**Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/BhattaSuyesh/optimizer-benchmark)

---

## What This Project Does

This tool lets you:

- **Train models** with different optimizers (SGD, Adam, AdamW, etc.) and hyperparameters side-by-side
- **Compare results** with auto-generated leaderboards, loss curves, memory charts, convergence analysis, and stability metrics
- **Make predictions** using trained models — draw digits, upload images, or predict California house prices on an interactive map
- **Export models** to ONNX format for deployment
- **Download PDF reports** summarizing benchmark results

---

## Supported Optimizers

| Optimizer | Description |
|-----------|-------------|
| **SGD** | Vanilla stochastic gradient descent |
| **SGD + Momentum** | SGD with momentum accumulation (default 0.9) |
| **Nesterov** | Nesterov accelerated gradient (look-ahead momentum) |
| **Adagrad** | Per-parameter adaptive learning rates |
| **RMSprop** | Exponential moving average of squared gradients |
| **Adam** | Combines momentum + RMSprop (most widely used) |
| **AdamW** | Adam with decoupled weight decay |

## Datasets & Models

| Dataset | Task | Model | Parameters |
|---------|------|-------|------------|
| **MNIST** | Digit classification (10 classes) | SmallCNN (Conv + BatchNorm + Dropout) | ~45k |
| **CIFAR-10** | Image classification (10 classes) | ResNet8 (residual blocks) | ~75k |
| **California Housing** | Median price regression | DeepMLP (4 hidden layers) | ~25k |

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Suyeshbatman/optimizer_comparison.git
cd optimizer_comparison
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app/streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## How to Use

### 1. Train & Benchmark

1. Select an experiment (MNIST, CIFAR-10, or California Housing)
2. Configure training: epochs, batch size, seeds (comma-separated for stability analysis)
3. Pick optimizers and learning rates to compare
4. Optionally enable: learning rate scheduler, early stopping, data augmentation
5. Click **Start Training** — watch live loss curves and metrics per run
6. After training, the app highlights the **Recommended Model** (best metric)

**Model naming convention:**
```
{optimizer}_lr{lr}_seed{seed}_ep{epochs}_{metric}{value}_{timestamp}.pt
```
Example: `adam_lr0.001_seed42_ep15_acc0.9912_20260515_143022.pt`

### 2. Compare Results

- **Leaderboard** — ranked table of all optimizer/lr combinations
- **Loss Curves** — training loss and test metric over epochs
- **Memory Comparison** — optimizer state size and peak GPU memory
- **Training Time** — wall-clock time per optimizer
- **Convergence Speed** — epochs to reach a target threshold
- **Stability Analysis** — metric variance across different seeds
- **Precision / Recall / F1** — per-class metrics for classification tasks
- **Regression Error** — MAE, MSE, bias analysis for housing predictions
- **Bias-Variance Analysis** — decomposition across seeds
- **PDF Report** — downloadable multi-page report with all charts and tables

### 3. Try the Model

- **MNIST** — draw a digit on a canvas or upload an image; see confidence per class
- **CIFAR-10** — upload any image; model classifies it into one of 10 categories
- **California Housing** — enter features manually, pick a preset (Suburban, Urban, Luxury, Rural), or select a location on an interactive map; get a predicted house price
- **Side-by-side comparison** — compare predictions from two different models on the same input
- **ONNX Export** — download the model in ONNX format for deployment

---

## Running Sweeps from the Command Line

For larger experiments, run sweeps directly without the UI:

```bash
python -m src.run_sweep --config configs/mnist_cnn.yaml
python -m src.run_sweep --config configs/cifar_resnet8.yaml
python -m src.run_sweep --config configs/housing_mlp.yaml
```

Results are saved to `runs/<experiment>/` and automatically appear in the Compare page.

---

## Project Structure

```
optimizer/
├── app/                        # Streamlit UI
│   ├── streamlit_app.py        # Entry point + sidebar navigation
│   └── pages/
│       ├── train_page.py       # Train & benchmark page
│       ├── compare_page.py     # Results comparison + PDF reports
│       └── predict_page.py     # Prediction UI + model management
├── src/                        # Core library
│   ├── datasets/               # MNIST, CIFAR-10, California Housing loaders
│   ├── models/                 # SmallCNN, ResNet8, DeepMLP architectures
│   ├── optim/                  # Optimizer registry
│   ├── bench/                  # Metrics, timing, logging, trainer loop
│   ├── run_sweep.py            # CLI sweep runner
│   └── plot_sweep.py           # Plot generation from results
├── configs/                    # YAML experiment configs
│   ├── mnist_cnn.yaml
│   ├── cifar_resnet8.yaml
│   └── housing_mlp.yaml
├── requirements.txt
└── README.md
```

---

## Training Tips

- **Use multiple seeds** (e.g., `42, 43, 44`) to get stability analysis on the Compare page
- **Enable data augmentation** for MNIST and CIFAR-10 to improve generalization
- **Adam/AdamW at lr=0.001** is a strong default for all three datasets
- **15+ epochs** for MNIST, **20+ epochs** for CIFAR-10, **30+ epochs** for Housing
- **Early stopping** with patience=5 prevents overfitting on longer runs
- **CosineAnnealing scheduler** often gives a small boost on CIFAR-10

---

## Tech Stack

- **PyTorch** — model training and inference
- **Streamlit** — interactive web UI
- **Matplotlib** — charts and PDF report generation
- **scikit-learn** — California Housing dataset, R² score, classification metrics
- **Folium** — interactive map for housing location selection
- **ONNX** — model export for deployment
