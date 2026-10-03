# Laptop Price Predictor

A machine learning project that predicts the price of a laptop from its specifications (brand, type, RAM, storage, screen, CPU, GPU, OS and weight). The whole workflow lives in a single Jupyter notebook: data cleaning, feature engineering, exploratory analysis, model comparison and model export.

## Highlights

- Cleans and engineers features from raw laptop listings (1,303 rows, 12 columns)
- Extracts useful signals from messy text columns: touchscreen, IPS panel, pixels per inch (PPI), CPU brand, split storage types (HDD / SSD) and GPU brand
- Trains and compares 11 regression models inside scikit-learn pipelines
- Best single model reaches an R² of about 0.89 on the held-out test set
- Exports the trained pipeline and the cleaned dataframe as pickle files for use in an app

## Repository Contents

| File | Description |
| --- | --- |
| `laptop-price-predictor.ipynb` | Full workflow: EDA, preprocessing, modelling, evaluation, export |
| `README.md` | Project overview |

The notebook expects a `laptop_data.csv` file in the same folder. It is not included in this repository, so add it before running. Running the final cells produces `df.pkl` and `pipe.pkl`.

## Dataset

The input has 12 columns: `Unnamed: 0` (index, dropped), `Company`, `TypeName`, `Inches`, `ScreenResolution`, `Cpu`, `Ram`, `Memory`, `Gpu`, `OpSys`, `Weight` and `Price`. There are no missing values and no duplicate rows.

## Workflow

### 1. Cleaning
- Dropped the index column
- Stripped the `GB` and `kg` units from `Ram` and `Weight` and converted them to numeric types

### 2. Feature engineering
- **Touchscreen** and **IPS**: binary flags parsed from `ScreenResolution`
- **PPI**: computed from the X and Y resolution and the screen size, then used in place of `Inches`, `X_res` and `Y_res`
- **Cpu brand**: grouped into Intel Core i3 / i5 / i7, Other Intel Processor and AMD Processor
- **Storage**: `Memory` split into HDD and SSD capacity columns; Hybrid and Flash Storage were dropped after checking their correlation with price
- **Gpu brand**: first token of the GPU name (Intel, Nvidia, AMD), with the single ARM row removed
- **OS**: grouped into Windows, Mac and Others/No OS/Linux

### 3. Target transform
The target is `log(Price)`, which gives a much more normal distribution. Predictions are therefore in log space and need `np.exp()` to convert back to a price.

### 4. Modelling
Data is split 85/15 with `random_state=2`. Each model sits in a pipeline with a `ColumnTransformer` that one-hot encodes the categorical columns (`Company`, `TypeName`, `Cpu brand`, `Gpu brand`, `os`).

## Results

Scores on the test set (target in log space):

| Model | R² | MAE |
| --- | --- | --- |
| Linear Regression | 0.807 | 0.210 |
| Ridge | 0.813 | 0.209 |
| Lasso | 0.807 | 0.211 |
| KNN | 0.802 | 0.193 |
| Decision Tree | 0.847 | 0.181 |
| SVR | 0.808 | 0.202 |
| Random Forest | 0.887 | 0.159 |
| Extra Trees | 0.875 | 0.160 |
| AdaBoost | 0.793 | 0.233 |
| Gradient Boosting | 0.882 | 0.159 |
| XGBoost | 0.881 | 0.165 |
| Stacking | 0.882 | 0.166 |
| **Voting (RF + GBDT + XGB + ET)** | **0.890** | **0.158** |

The Voting Regressor performs best and is the model exported at the end of the notebook.

## Tech Stack

Python 3, pandas, NumPy, matplotlib, seaborn, scikit-learn, XGBoost

## Getting Started

```bash
git clone https://github.com/sudhanshukhare18/laptop-price-predictor.git
cd laptop-price-predictor

pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter

# place laptop_data.csv in this folder, then:
jupyter notebook laptop-price-predictor.ipynb
```

Note: the notebook was written with older scikit-learn (it uses `OneHotEncoder(sparse=False)`). On scikit-learn 1.2 or newer, change that argument to `sparse_output=False`.

## Using the Exported Model

```python
import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Build a row with the same columns as df (minus Price), in the same order:
# Company, TypeName, Ram, Weight, Touchscreen, Ips, ppi, Cpu brand, HDD, SSD, Gpu brand, os
query = pd.DataFrame([{
    'Company': 'Dell', 'TypeName': 'Notebook', 'Ram': 8, 'Weight': 2.0,
    'Touchscreen': 0, 'Ips': 1, 'ppi': 141.2, 'Cpu brand': 'Intel Core i5',
    'HDD': 0, 'SSD': 256, 'Gpu brand': 'Intel', 'os': 'Windows'
}])

price = np.exp(pipe.predict(query))[0]
print(round(price))
```

## Possible Improvements

- Add a web front end (for example Streamlit or Flask) on top of `pipe.pkl`
- Tune hyperparameters with cross-validation instead of a single train/test split
- Add `laptop_data.csv` and a `requirements.txt` to make the project reproducible out of the box

## Author

Sudhanshu Khare
[GitHub](https://github.com/sudhanshukhare18) · [LinkedIn](https://linkedin.com/in/sudhanshu-khare)
