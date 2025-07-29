import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")

st.title("Customer Segmentation 👥")
st.markdown("This page uses K-Means clustering to group customers into distinct segments based on their tenure, monthly, and total charges.")

# Load clustered data
try:
    df_clustered = joblib.load('clustered_data.pkl')
except FileNotFoundError:
    st.error("Clustered data not found. Please run the `train_model.py` script first.")
    st.stop()

# --- 3D Scatter Plot Visualization ---
st.header("Interactive 3D Cluster Visualization")

fig_3d = px.scatter_3d(df_clustered,
                       x='tenure',
                       y='MonthlyCharges',
                       z='TotalCharges',
                       color='Cluster',
                       symbol='Cluster',
                       hover_name='customerID',
                       hover_data={'Cluster': True, 'Churn': True},
                       title="Customer Segments in 3D Space")
fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
st.plotly_chart(fig_3d, use_container_width=True)


# --- Cluster Analysis ---
st.header("Analysis of Customer Segments")
st.markdown("Let's analyze the characteristics of each customer segment.")

# Calculate statistics for each cluster
cluster_analysis = df_clustered.groupby('Cluster').agg(
    Customer_Count=('customerID', 'count'),
    Avg_Tenure=('tenure', 'mean'),
    Avg_Monthly_Charges=('MonthlyCharges', 'mean'),
    Churn_Rate=('Churn', lambda x: (x == 'Yes').mean())
).reset_index()

# Format the analysis table
cluster_analysis['Avg_Tenure'] = cluster_analysis['Avg_Tenure'].map('{:.1f} months'.format)
cluster_analysis['Avg_Monthly_Charges'] = cluster_analysis['Avg_Monthly_Charges'].map('${:.2f}'.format)
cluster_analysis['Churn_Rate'] = cluster_analysis['Churn_Rate'].map('{:.2%}'.format)

# Define segment personas based on analysis
personas = {
    0: "💎 High-Value, Loyal",
    1: "🌱 New, Low-Spending",
    2: "💸 Mid-Value, Moderate Tenure",
    3: "🚨 At-Risk, High-Spending"
}
# This mapping is an example; you should adjust it based on your analysis of the table below.
# A good practice is to run the app, view the table, then define the personas.
cluster_analysis['Persona (Example)'] = cluster_analysis['Cluster'].map(personas)


st.dataframe(cluster_analysis, use_container_width=True)

st.info("""
**How to interpret the segments (Example):**
- **💎 High-Value, Loyal:** Customers with long tenure and high total charges, but low churn rate. These are your best customers.
- **🌱 New, Low-Spending:** Recent customers with low tenure and low monthly charges. A key group to nurture.
- **💸 Mid-Value, Moderate Tenure:** The average customer. Opportunities exist to upsell or increase loyalty.
- **🚨 At-Risk, High-Spending:** Customers with high monthly bills but may have a higher churn rate. These need immediate attention.
""", icon="💡")