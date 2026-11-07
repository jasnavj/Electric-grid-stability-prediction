# Electric Grid Stability Prediction

A supervised machine learning project that predicts whether a simulated electrical grid is **stable** or **unstable** from 12 grid-related input features. The project includes exploratory data analysis, model comparison, SVC hyperparameter tuning, model evaluation, and an interactive Streamlit application.

> **Educational use only:** This application uses simulated data. It is not intended for real-time grid monitoring, protection, or operational decisions.

## Project highlights

- Dataset: UCI Electrical Grid Stability Simulated Data
- Observations: 10,000
- Task: Binary classification (`stable` / `unstable`)
- Input features: `tau1`–`tau4`, `p1`–`p4`, `g1`–`g4`
- Final estimator: tuned Support Vector Classifier (SVC), RBF kernel
- Test accuracy: **97.20%**
- Best cross-validation macro F1 during tuning: **95.59%**
- Best parameters: `C=10`, `kernel='rbf'`, `gamma='auto'`

## Repository structure

```text
electric-grid-stability-prediction/
├── app.py
├── notebooks/
│   └── Project_electrical_grid_stability.ipynb
├── svc_model.sav
├── svc_scaler.sav
├── requirements.txt
├── .gitignore
└── README.md
```

The model and scaler files are kept in the repository root because `app.py` loads them from the same directory as the application.

## Dataset

Download the **Electrical Grid Stability Simulated Data** dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data

Place the downloaded CSV file named `Electric_Grid_Stability_.csv` in the repository root before running the notebook. The notebook's data-loading cell uses this relative filename.

## Run the Streamlit application locally

1. Clone this repository:

   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd electric-grid-stability-prediction
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

The app provides an overview, a stability predictor, model performance information, and project methodology. Enter values for the 12 input features and select **Predict stability**.

## Model evaluation

The notebook reports a held-out test accuracy of **97.20%** on 2,000 test observations. The classification report gives rounded class-wise precision, recall, and F1-score of 0.96 for the stable class and 0.98 for the unstable class. The tuned SVC achieved a best 5-fold cross-validation macro F1 score of approximately **0.9559** during hyperparameter search.

## Tools and libraries

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost (used in model comparison in the notebook)
- Streamlit

## Notes

- The saved SVC and scaler were serialized with **scikit-learn 1.9.1**. Use the version pinned in `requirements.txt` when loading the saved artifacts.
- Only load pickle files from sources you trust.
- The notebook expects the UCI CSV to be downloaded separately; the dataset is not included in this repository.

## Acknowledgements

Dataset: UCI Machine Learning Repository.  
Developed as a machine learning training project.

## Scope and Limitations

This project is an educational Machine Learning application developed using simulated electrical grid stability data.

It demonstrates how machine learning can be used to classify electrical grid operating conditions as Stable or Unstable based on 12 input parameters.

**Important:** This application is intended for learning and demonstration purposes only. It is not connected to a real electrical power grid and should not be used for real-time grid monitoring, operational control, or critical infrastructure decisions.
