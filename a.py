import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error

df=pd.read_csv(r"C:\Users\NTC\Desktop\laptop_battery_health_usage.csv")

df_clean=df.drop("device_id",axis=1)
x=df_clean.drop("battery_health_percent",axis=1)
y=df_clean["battery_health_percent"]

cat_col=["brand","os","usage_type","overheating_issues"]
numerical_col=x.columns.difference(cat_col)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), cat_col),
        ("num", "passthrough", numerical_col)
    ]
)
#===================================================================================================
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train_processed=preprocessor.fit_transform(x_train)
x_test_processed=preprocessor.transform(x_test)

l_model=LinearRegression()
l_model.fit(x_train_processed,y_train)

l_pred=l_model.predict(x_test_processed)

l_mse = mean_squared_error(y_test, l_pred)
l_rmse = root_mean_squared_error(y_test, l_pred)
l_r2 = r2_score(y_test, l_pred)

print(f"MSE: {l_mse:.2f}")
print(f"RMSE: {l_rmse:.2f}")
print(f"R² Score: {l_r2:.2f}")
#=========================================================================================================

r_model=RandomForestRegressor(random_state=42,n_estimators=100)
r_model.fit(x_train_processed,y_train)
r_pred=r_model.predict(x_test_processed)

rf_mse = mean_squared_error(y_test, r_pred)
rf_rmse = root_mean_squared_error(y_test, r_pred)
rf_r2 = r2_score(y_test, r_pred)
print('================================================')
print(f"MSE: {rf_mse:.2f}")
print(f"RMSE: {rf_rmse:.2f}")
print(f"R² Score: {rf_r2:.2f}")
#==========================================================================================================

comparison=pd.DataFrame({
    "algo":['Linear regression','random forest'],
    'MSE':[l_mse,rf_mse],
    'RMSE':[l_rmse,rf_rmse],
    'r2':[l_r2,rf_r2]
})
print('================================================')
print(comparison)
print('================================================')
#===========================================================================================================
plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)
plt.scatter(y_test, l_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Linear Regression\nPredicted vs Actual')
plt.xlabel('Actual Value')
plt.ylabel('Predicted value')

plt.subplot(2, 2, 2)
plt.scatter(y_test, r_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Random Forest\nPredicted vs Actual')
plt.xlabel('Actual Value')
plt.ylabel('Predicted value')

plt.subplot(2, 2, 3)
feature_names = preprocessor.get_feature_names_out()
importances = r_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title('importane of the feature(Random Forest)')
plt.gca().invert_yaxis()

plt.subplot(2, 2, 4)
df_for_corr = df_clean.copy()
for col in cat_col:
    df_for_corr[col] = df_for_corr[col].astype('category').cat.codes
sns.heatmap(df_for_corr.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('map of association between feature')

plt.tight_layout()
plt.savefig('results_visualization.png', dpi=300, bbox_inches='tight')  
plt.show()
