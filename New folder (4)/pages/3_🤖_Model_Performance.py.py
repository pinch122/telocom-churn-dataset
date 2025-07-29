import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")

st.title("Churn Prediction Model Performance 🤖")
st.markdown("This page evaluates the performance of the trained Logistic Regression model on the unseen test dataset.")

# Load model and test data
try:
    model = joblib.load('churn_model_pipeline.pkl')
    X_test, y_test = joblib.load('test_data.pkl')
except FileNotFoundError:
    st.error("Model or test data not found. Please run the `train_model.py` script first.")
    st.stop()

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# --- Display Performance Metrics ---
st.header("Classification Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
col2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.2%}")

# --- Display Visualizations ---
st.header("Performance Visualizations")
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, 
                        labels=dict(x="Predicted Label", y="True Label"),
                        x=['Not Churn', 'Churn'], y=['Not Churn', 'Churn'],
                        color_continuous_scale='Blues',
                        title="Confusion Matrix")
    fig_cm.update_layout(title_x=0.5)
    st.plotly_chart(fig_cm, use_container_width=True)

with viz_col2:
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dash')))
    fig_roc.update_layout(title="Receiver Operating Characteristic (ROC) Curve",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          title_x=0.5)
    st.plotly_chart(fig_roc, use_container_width=True)

# --- Feature Importance ---
st.header("Model Feature Importance")
st.markdown("Feature importance shows which factors have the biggest impact on the churn prediction.")

# Extract feature names and coefficients
try:
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    
    cat_features = preprocessor.transformers_[1][1].get_feature_names_out()
    num_features = preprocessor.transformers_[0][2]
    
    all_features = list(num_features) + list(cat_features)
    coefficients = classifier.coef_[0]

    feature_importance = pd.DataFrame({'Feature': all_features, 'Importance': coefficients})
    feature_importance['Absolute Importance'] = feature_importance['Importance'].abs()
    feature_importance = feature_importance.sort_values(by='Absolute Importance', ascending=False).head(15)

    fig_importance = px.bar(feature_importance, 
                            x='Importance', 
                            y='Feature', 
                            orientation='h',
                            title='Top 15 Most Important Features',
                            color='Importance',
                            color_continuous_scale='RdBu_r')
    fig_importance.update_layout(title_x=0.5, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_importance, use_container_width=True)

except Exception as e:
    st.warning(f"Could not extract feature importances. Error: {e}")