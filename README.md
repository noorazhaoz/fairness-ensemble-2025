# Fairness-Aware Post-Processing Tasks

This repository provides the implementation for the ensemble-based post-processing framework developed in our work. The code includes several Mixture and Mixture-of-Experts (MoE) perturbation models, together with utilities for data loading, evaluation, and running quick experiments.

## 📌 Features

- Mixture model post-processing  
- Mixture-of-Experts (MoE) post-processing  
- Single-pretrained and Two-pretrained variants  
- Supports performance–fairness trade-off experiments  
- Quickstart demo for the Adult dataset  

---

## 📂 Repository Structure

```
fair-postproc-tasks/
│
├── algorithms/                 # Core post-processing algorithms
│   ├── mixture_one_pretrained.py
│   ├── mixture_two_pretrained.py
│   ├── moe_one_pretrained.py
│   ├── moe_two_pretrained.py
│   └── __init__.py
│
├── utils/                      # Helper functions
│   ├── data_loader.py
│   ├── common.py
│   └── __init__.py
│
├── metrics/                    # Fairness / performance metrics
│   └── (add custom metrics if needed)
│
├── data/                       # Dataset interface
│   └── __init__.py
│
├── demo/                       # Quickstart examples
│   ├── quickstart_adult.py
│   └── __init__.py
│
├── main.py                     # Main entry point for running experiments
└── requirements.txt            # Python dependencies
```

---

## 📦 Installation

### 1. (Optional) Create a virtual environment

```bash
python3 -m venv env
source env/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Quickstart Example

Run a simple experiment using the Adult dataset:

```bash
python demo/quickstart_adult.py
```

Or run the main script with custom parameters:

```bash
python main.py --dataset adult --lambda 0.1 --model mixture_two
```

Available `--model` options:

- `mixture_one`
- `mixture_two`
- `moe_one`
- `moe_two`

---

## 🧩 Algorithm Summary

### Mixture Models
Use a **global scalar weight** to combine predictions from performance and fairness models.

### Mixture-of-Experts (MoE)
Use a **gating network** (e.g., logistic regression) to learn instance-specific weights.

Both support:

- **One-pretrained** version  
- **Two-pretrained** version  

---

## 📊 Datasets

