import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import ipywidgets as widgets
from IPython.display import display
import streamlit as st
from sklearn.metrics import roc_curve

# Updated predictor list (Removed TSH_Code)
predictors = ['HTN_code', 'HDL_code', 'HOMA2B_code', 'HOMA2IR_code', 'BMI_code', 'HsCRP_code']
X = data[predictors]
y = data['Diabetestype_code']

# Add constant for logistic regression
X = sm.add_constant(X)

# Fit logistic regression model
model = sm.Logit(y, X).fit()
print(model.summary())

# Extract coefficients (excluding the constant)
coefs = model.params[1:]
coefs_std = (coefs - coefs.min()) / (coefs.max() - coefs.min()) * 100

# Compute optimal risk threshold using ROC curve
y_pred_prob = model.predict(X)
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Probability Cut-off for Risk Categories: {optimal_threshold:.2f}")

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

# Interactive UI with ipywidgets
htn = widgets.IntSlider(min=1, max=2, step=1, value=1, description="HTN_code")
hdl = widgets.IntSlider(min=1, max=2, step=1, value=1, description="HDL_code")
homa2b = widgets.IntSlider(min=1, max=2, step=1, value=1, description="HOMA2B_code")
homa2ir = widgets.IntSlider(min=1, max=2, step=1, value=1, description="HOMA2IR_code")
bmi = widgets.IntSlider(min=1, max=2, step=1, value=1, description="BMI_code")
hscrp = widgets.IntSlider(min=1, max=2, step=1, value=1, description="HsCRP_code")
output = widgets.Output()

def update_risk(change):
    with output:
        output.clear_output()
        total_points, probability, risk_category = predict_diabetes_risk(
            htn.value, hdl.value, homa2b.value, homa2ir.value, bmi.value, hscrp.value
        )
        print(f"Total Nomogram Points: {total_points:.1f}")
        print(f"Predicted Diabetes Risk: {probability*100:.2f}%")
        print(f"Risk Category: {risk_category}")

for widget in [htn, hdl, homa2b, homa2ir, bmi, hscrp]:
    widget.observe(update_risk, names='value')

display(htn, hdl, homa2b, homa2ir, bmi, hscrp, output)
update_risk(None)

# Streamlit Web App
st.title("Diabetes Risk Calculator")
st.sidebar.header("Enter Predictor Values")
htn = st.sidebar.slider("HTN Code", 1, 2, 1)
hdl = st.sidebar.slider("HDL Code", 1, 2, 1)
homa2b = st.sidebar.slider("HOMA2B Code", 1, 2, 1)
homa2ir = st.sidebar.slider("HOMA2IR Code", 1, 2, 1)
bmi = st.sidebar.slider("BMI Code", 1, 2, 1)
hscrp = st.sidebar.slider("HsCRP Code", 1, 2, 1)
total_points, probability, risk_category = predict_diabetes_risk(htn, hdl, homa2b, homa2ir, bmi, hscrp)
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
low_threshold = optimal_threshold * 0.5 * 100
moderate_threshold = optimal_threshold * 100
plt.axvline(low_threshold, color='green', linestyle='--', label="Low Risk Threshold")
plt.axvline(moderate_threshold, color='orange', linestyle='--', label="Moderate Risk Threshold")
plt.axvline(100, color='red', linestyle='--', label="High Risk Threshold")
plt.legend()
plt.show()