import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

# Load and clean data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    # Create a copy for clustering before encoding 'Churn'
    df_for_clustering = df.copy()
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df, df_for_clustering

df, df_for_clustering = load_data()

# --- 1. Train and Save the Churn Prediction Model ---

# Define features (X) and target (y)
X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create the full pipeline with preprocessing and logistic regression model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the trained pipeline and the test data for evaluation
joblib.dump(model_pipeline, 'churn_model_pipeline.pkl')
joblib.dump((X_test, y_test), 'test_data.pkl')

print("✅ Churn prediction model trained and saved.")
print(f"✅ Test data saved for performance evaluation.")


# --- 2. Train and Save the Customer Segmentation Model ---

# Select features for clustering
features_for_clustering = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_cluster = df_for_clustering[features_for_clustering]

# Scale the features
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Train a K-Means model
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_cluster_scaled)

# Save the K-Means model and the original data with cluster labels
df_for_clustering['Cluster'] = kmeans.labels_
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(df_for_clustering, 'clustered_data.pkl')
joblib.dump(scaler, 'kmeans_scaler.pkl')

print("✅ Customer segmentation model trained and saved.")
print("✅ Clustered data saved for analysis.")