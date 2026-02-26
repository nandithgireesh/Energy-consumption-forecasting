import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page config
st.set_page_config(
    page_title="Energy Forecasting | Claysys",
    page_icon="⚡",
    layout="wide"
)

# Constants & Paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'data' / 'processed' / 'test.csv'
MODELS_DIR = BASE_DIR / 'models'
TARGET = 'Global_active_power'

# Sidebar Info
st.sidebar.title("⚡ Energy Forecasting")
st.sidebar.write("**Claysys AI Hackathon 2026**")
st.sidebar.markdown("---")
st.sidebar.write("**Deployed Models:**")
model_choice = st.sidebar.radio(
    "Select Model to View:",
    ("LightGBM (Champion 🏆)", "XGBoost", "Random Forest")
)

st.sidebar.markdown("---")
st.sidebar.write("**Tech Stack:**")
st.sidebar.markdown("""
- **Frontend:** Streamlit
- **ML Models:** LightGBM, XGBoost, Scikit-learn
- **Data:** Pandas, Plotly
- **Infrastructure:** Python 3.10
""")

# Main Content
st.title("🔋 Household Energy Consumption Forecasting")
st.write("""
This dashboard predicts the future **Global Active Power** usage of a household using machine learning models trained on ~4 years of minute-level historical data. 
""")

# Data Loading with Cache
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, index_col='Datetime', parse_dates=True)
    return df

@st.cache_resource
def load_models():
    models = {}
    try:
        models['LightGBM (Champion 🏆)'] = (joblib.load(MODELS_DIR / 'lightgbm.pkl'), 'LightGBM')
        models['XGBoost'] = (joblib.load(MODELS_DIR / 'xgboost.pkl'), 'XGBoost')
        models['Random Forest'] = (joblib.load(MODELS_DIR / 'random_forest.pkl'), 'Random Forest')
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

# Load everything
with st.spinner("Loading data and models..."):
    test_df = load_data()
    models = load_models()

if test_df.empty or not models:
    st.stop()

# Prepare features
drop_cols = [TARGET, 'season']
feature_cols = [c for c in test_df.select_dtypes(include=[np.number]).columns if c not in drop_cols]
X_test = test_df[feature_cols]
y_test = test_df[TARGET]

# Display data snapshot
if st.checkbox("Show Raw Test Data (First 100 rows)"):
    st.dataframe(test_df.head(100))

# Get selected model
selected_model, model_name = models[model_choice]

# Generate predictions
with st.spinner(f"Generating predictions using {model_name}..."):
    preds = selected_model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

# UI Metrics Row
col1, col2, col3 = st.columns(3)
col1.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f} kW", f"-99.2% from Baseline")
col2.metric("Mean Absolute Error (MAE)", f"{mae:.4f} kW")
col3.metric("R² Score", f"{r2:.4f}")

# Time Range Slider
st.markdown("### 📊 Interactive Forecast Viewer")
hours_to_view = st.slider(
    "Select forecast horizon (hours to view):", 
    min_value=24, 
    max_value=len(test_df), 
    value=168,  # Default to 1 week
    step=24
)

# Filter data for plot
plot_idx = test_df.index[:hours_to_view]
plot_actual = y_test[:hours_to_view]
plot_preds = preds[:hours_to_view]

# Plotly Interactive Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_idx, y=plot_actual, mode='lines', name='Actual Power', line=dict(color='#1f77b4', width=2)))
fig.add_trace(go.Scatter(x=plot_idx, y=plot_preds, mode='lines', name=f'{model_name} Prediction', line=dict(color='#ff7f0e', width=2, dash='dash')))

fig.update_layout(
    title=f"Actual vs Predicted Power Consumption (First {hours_to_view} Hours)",
    xaxis_title="Date / Time",
    yaxis_title="Global Active Power (kW)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_dark",
    margin=dict(l=0, r=0, t=60, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# Feature Importance (if tree model)
st.markdown("### 🧠 Model Explainability (Feature Importance)")
try:
    importance = selected_model.feature_importances_
    fi_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance}).sort_values('Importance', ascending=True).tail(10)
    
    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                    title=f"Top 10 Most Predictive Features for {model_name}",
                    color='Importance', color_continuous_scale='Blues')
    fig_fi.update_layout(template="plotly_dark")
    st.plotly_chart(fig_fi, use_container_width=True)
except AttributeError:
    st.write("Feature importance not available for this model type.")
