import os
st.write(f"Current Working Directory: {os.getcwd()}")
st.write(f"Files in Directory: {os.listdir()}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from sklearn.metrics import roc_curve

# Load dataset (Make sure 'data.csv' is in the same GitHub repo)
@st.cache
def load_data():
    return pd.read_excel("./data.xlsx")

data = load_data()

# Updated predictor list (Removed TSH_Code)
predictors = ['HTN_code', 'HDL_code', 'HOMA2B_code', 'HOMA2IR_code', 'BMI_code', 'HsCRP_code']
X = data[predictors]
y = data['Diabetestype_code']

# Add constant for logistic regression
X = sm.add_constant(X)

# Fit logistic regression model
model = sm.Logit(y, X).fit()

# Extract coefficients
coefs = model.params[1:]
coefs_std = (coefs - coefs.min()) / (coefs.max() - coefs.min()) * 100

# Compute optimal risk threshold using ROC curve
y_pred_prob = model.predict(X)
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]

# Define risk categories
def predict_diabetes_risk(htn, hdl, homa2b, homa2ir, bmi, hscrp):
    input_data = np.array([1, htn, hdl, homa2b, homa2ir, bmi, hscrp])
    total_points = np.sum(input_data[1:] * coefs_std.values)
    logit_score = np.dot(input_data, model.params)
    probability = 1 / (1 + np.exp(-logit_score))

    if probability < optimal_threshold * 0.5:
        risk_category = "Low Risk"
    elif probability < optimal_threshold:
        risk_category = "Moderate Risk"
    else:
        risk_category = "High Risk"

    return total_points, probability, risk_category

# Streamlit Web App UI
st.title("Diabetes Risk Calculator")
st.sidebar.header("Enter Predictor Values")

# User Input
htn = st.sidebar.slider("HTN Code", 1, 2, 1)
hdl = st.sidebar.slider("HDL Code", 1, 2, 1)
homa2b = st.sidebar.slider("HOMA2B Code", 1, 2, 1)
homa2ir = st.sidebar.slider("HOMA2IR Code", 1, 2, 1)
bmi = st.sidebar.slider("BMI Code", 1, 2, 1)
hscrp = st.sidebar.slider("HsCRP Code", 1, 2, 1)

# Calculate Risk
total_points, probability, risk_category = predict_diabetes_risk(htn, hdl, homa2b, homa2ir, bmi, hscrp)

# Display Results
st.subheader("Prediction Results")
st.write(f"**Total Nomogram Points:** {total_points:.1f}")
st.write(f"**Predicted Diabetes Risk:** {probability*100:.2f}%")
st.write(f"**Risk Category:** {risk_category}")

# Nomogram with Threshold Markers
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
sns.barplot(x=coefs_std, y=predictors, palette='coolwarm', ax=ax)
ax.set_xlabel('Nomogram Points (Scaled)', fontsize=14, fontweight='bold')
ax.set_ylabel('Predictors', fontsize=14, fontweight='bold')
ax.set_title('Nomogram for Diabetes Risk', fontsize=16, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.6)

# Threshold Markers
low_threshold = optimal_threshold * 0.5 * 100
moderate_threshold = optimal_threshold * 100
plt.axvline(low_threshold, color='green', linestyle='--', label="Low Risk Threshold")
plt.axvline(moderate_threshold, color='orange', linestyle='--', label="Moderate Risk Threshold")
plt.axvline(100, color='red', linestyle='--', label="High Risk Threshold")
plt.legend()

# Display the plot in Streamlit
st.pyplot(fig)
