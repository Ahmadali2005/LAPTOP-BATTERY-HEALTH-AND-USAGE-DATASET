🔋 Laptop Battery Health Prediction

A complete machine learning pipeline to predict laptop battery health percentage using real-world usage data.






📌 Overview

This project predicts battery_health_percent using laptop usage and hardware behavior data.
It demonstrates a full end-to-end ML workflow:

Data → Preprocessing → Modeling → Evaluation → Visualization

📊 Dataset

Dataset from Kaggle
Target: battery_health_percent
Task: Regression

🧱 Project Structure
├── data/
│   └── battery_data.csv
├── notebooks/
│   └── exploration.ipynb
├── models/
│   └── trained_model.pkl
├── train_model.py
├── requirements.txt
└── README.md

⚙️ Tech Stack

Python

pandas, numpy

scikit-learn

matplotlib, seaborn

🔧 Preprocessing

Missing value handling

Feature scaling

Categorical encoding with:

ColumnTransformer

OneHotEncoder

🤖 Models
Model	Purpose
Linear Regression	Baseline
Random Forest Regressor	Main predictor
📈 Evaluation Metrics

MSE

RMSE

R² Score

🏆 Results

✔ Random Forest achieved significantly higher R²
✔ Lower prediction error than Linear Regression
✔ Key features:

Cycle count

Average temperature

Daily usage hours

▶️ Installation
git clone <your-repo-url>
cd laptop-battery-health-prediction
pip install -r requirements.txt

🚀 Usage
python train_model.py


or explore:

jupyter notebook

📊 Sample Outputs

Predicted vs Actual plot
<img width="475" height="80" alt="image" src="https://github.com/user-attachments/assets/b410c5c1-3d46-41c2-a038-340cd00f92cd" />


Feature importance chart

Correlation heatmap

<img width="1906" height="949" alt="image" src="https://github.com/user-attachments/assets/28946cc3-4689-414c-98f7-17ca7d7d9325" />


🔮 Future Improvements

Hyperparameter tuning

Try XGBoost / LightGBM

Model deployment (API or dashboard)

Real-time battery monitoring
