import pandas as pd
import streamlit as st
import plotly.express as px

# Page Configuration 
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="💔",
    layout="wide"
)

# Function to convert df to csv
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Data Loading and Cleaning ---
@st.cache_data 
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True) 
    return df

df = load_data()

# --- Main Page Title ---
st.title("Telco Customer Churn Dashboard 📊")
st.markdown("This dashboard provides an analysis of customer churn based on different segments.")

# --- Sidebar Filters ---
st.sidebar.header("Customer Segments")

# Filter by Contract
selected_contract = st.sidebar.multiselect(
    "Select Contract Type",
    options=df['Contract'].unique(),
    default=df['Contract'].unique()
)

# Filter by Internet Service
selected_internet = st.sidebar.multiselect(
    "Select Internet Service",
    options=df['InternetService'].unique(),
    default=df['InternetService'].unique()
)

# Filter the dataframe based on selection
filtered_df = df[
    df['Contract'].isin(selected_contract) &
    df['InternetService'].isin(selected_internet)
]

# --- Main Content ---
if filtered_df.empty:
    st.warning("No data available for the selected filters. Please select at least one option for each filter.")
else:
    # KPI Section 
    st.header("Segment Performance")
    col1, col2, col3 = st.columns(3)

    # Metric 1: Churn Rate
    churn_rate = (filtered_df['Churn'].value_counts(normalize=True).get('Yes', 0)) * 100
    col1.metric("Churn Rate", f"{churn_rate:.2f}%", delta=f"{churn_rate - 26.54:.2f}% vs. Avg.", delta_color="inverse")

    # Metric 2: Total Customers in Segment
    total_customers = filtered_df.shape[0]
    col2.metric("Total Customers", f"{total_customers}")

    # Metric 3: Monthly Revenue Lost to Churn
    revenue_lost = filtered_df[filtered_df['Churn'] == 'Yes']['MonthlyCharges'].sum()
    col3.metric("Monthly Revenue Lost", f"${revenue_lost:,.2f}")

    # Charts Section 
    st.header("Visual Analysis")
    
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Chart 1: Churn by Payment Method
        fig_payment = px.bar(
            filtered_df, x='PaymentMethod', color='Churn', title='Churn by Payment Method',
            barmode='group', labels={'PaymentMethod': 'Payment Method', 'count': 'Number of Customers'},
            color_discrete_map={'No': '#1f77b4', 'Yes': '#d62728'}
        )
        st.plotly_chart(fig_payment, use_container_width=True)

    with col_chart2:
        # Chart 2: Churn by Tenure
        fig_tenure = px.histogram(
            filtered_df, x='tenure', color='Churn', title='Churn by Customer Tenure (Months)',
            marginal='box', labels={'tenure': 'Tenure (Months)'},
            color_discrete_map={'No': '#1f77b4', 'Yes': '#d62728'}
        )
        st.plotly_chart(fig_tenure, use_container_width=True)
    
    # --- Data Downloader ---
    st.header("Download Filtered Data")
    st.markdown("Download the data from your current selection as a CSV file.")
    csv = convert_df_to_csv(filtered_df)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name='filtered_churn_data.csv',
        mime='text/csv',
    )