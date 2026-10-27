# Heart Failure Mortality Prediction — Cost-Aware Neural Network

Predicts 30-day mortality from heart failure using a multilayer perceptron (MLP)
trained on the UCI Heart Failure Clinical Records dataset (n=299). Four experiments
systematically evaluate design choices relevant to clinical risk modeling.

## Experiments

| # | Question | Method |
|---|----------|--------|
| 1 | Does minority oversampling improve recall? | Baseline vs. SMOTE-style oversampling |
| 2 | Can cost-minimizing thresholds reduce clinical cost? | Tuned τ vs. default 0.5 at FN:FP ratios of 5, 10, 20 |
| 3 | Which regularizer best controls overfitting? | No regularization vs. L2 (λ grid) vs. Dropout (p grid) |
| 4 | Does feature standardization affect convergence? | Standardized vs. raw continuous inputs |

Each condition is evaluated under stratified 5-fold CV with paired t-tests
for significance testing across folds.

## Architecture

- Input: 11 clinical features (6 continuous, 5 binary)
- Hidden layers: 32 → 16 (ReLU activations)
- Output: single logit (BCEWithLogitsLoss)
- Optimizer: Adam (lr=1e-3), early stopping on validation loss (patience=20)

## Key Results

- Cost-tuned thresholds track the theoretical optimum τ* = 1/(1+r) across cost ratios
- Oversampling improves recall at the cost of precision — tradeoff depends on clinical cost ratio
- L2 regularization reduces train/val gap more consistently than dropout at this dataset size
- Feature standardization substantially accelerates convergence (earlier best epoch)

## Stack

Python · PyTorch · Scikit-learn · NumPy · Pandas · Matplotlib · SciPy

## Data

UCI ML Repository — Heart Failure Clinical Records (ID 519)  
Loaded automatically via `ucimlrepo`.

## Usage

Open `Heart_Failure_NN.ipynb` in Google Colab or Jupyter.  
All dependencies install automatically. Results export to `results/` as CSVs.
