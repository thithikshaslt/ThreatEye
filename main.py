import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score

st.title("Intrusion Detection System using Machine Learning")

st.header("1. Load Dataset")
st.write("Upload your CSV file, or click the link below to download a sample file.")

sample_files = [
    {'name': 'Sample Data 1', 'url': 'https://drive.google.com/file/d/1_gRC0A34RnnJZG3M3GiBZViVvn0-ATbJ/view?usp=sharing'},
    {'name': 'Sample Data 2', 'url': 'https://drive.google.com/file/d/1XT7Av59Ot04MrICRTpeyDT5L1dDs6lmI/view?usp=sharing'},
    {'name': 'Sample Data 3', 'url': 'https://drive.google.com/file/d/13VkgTdO-Y5N7cPadetdbpDhUWVh6eIco/view?usp=sharing'}
]

table = "| Sample File Name | Download Link |\n"
table += "|------------------|---------------|\n"
for file in sample_files:
    table += f"| {file['name']} | [Download]({file['url']}) |\n"

st.markdown(table)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = data.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], errors='ignore')
    
    required_columns =  ['Src Port', 'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts',
       'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
       'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
       'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
       'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
       'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
       'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
       'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
       'Bwd IAT Min', 'Bwd PSH Flags', 'Fwd Header Len', 'Bwd Header Len',
       'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max',
       'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'SYN Flag Cnt',
       'PSH Flag Cnt', 'ACK Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
       'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Subflow Fwd Pkts',
       'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
       'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Active Mean', 'Active Std',
       'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
       'Idle Min']
    
    missing_cols = [col for col in required_columns if col not in data.columns]
    
    if missing_cols:
        st.error(f"Dataset is missing the following columns: {missing_cols}")
    else:
        st.write("Dataset Preview:")
        st.dataframe(data.head())
        
        st.header("Data Preprocessing")
        
        # numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        # data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        # Identify numeric columns
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

        # Replace infinite values with NaN
        data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Fill missing values with column mean
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

        # Clip extreme values to avoid overflow errors
        data[numeric_cols] = data[numeric_cols].clip(lower=-1e9, upper=1e9)

        # Debugging: Check for remaining issues
        if np.isinf(data[numeric_cols]).values.any():
            st.error("Dataset still contains infinite values after processing!")

        if data[numeric_cols].isna().values.any():
            st.error("Dataset still contains NaN values after processing!")

        st.write("Preprocessed Data Preview:")
        st.dataframe(data.head())
        
        st.header("Feature Scaling")
        X = data.drop(columns=['Label'], errors='ignore')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        st.write("Done feature scaling.")
        
        st.header("Querying the models")
        models = ["Logistic Regression", "Decision Tree", "Random Forest", "KNN"]
        model_metrics = {}
        
        for model_name in models:
            st.subheader(f"{model_name}")
            model_path = f"models/{model_name.replace(' ', '_')}_model.pkl"
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                y_pred = model.predict(X_scaled)
                data[f'{model_name}_Label'] = ['Anomaly' if x == 1 else 'Normal' for x in y_pred]
                
                # Cross-validation
                scores = cross_val_score(model, X_scaled, y_pred, cv=5, scoring='accuracy')
                model_metrics[model_name] = scores.mean()
            else:
                st.error(f"Model file missing: {model_path}")
        
        # Majority prediction
        data['Majority prediction'] = data[[f'{m}_Label' for m in models]].mode(axis=1)[0]
        
        st.header("Model Metrics Summary")
        st.write(pd.DataFrame(model_metrics, index=['Accuracy']))
        
        st.header("Final Output")
        malicious_counts = (data['Majority prediction'] == 'Anomaly').sum()
        total_records = len(data)
        malicious_percentage = (malicious_counts / total_records) * 100 if total_records > 0 else 0
        
        st.write(f"Total Records: {total_records}")
        st.write(f"Malicious Records: {malicious_counts}")
        st.write(f"Percentage of Malicious Records: {malicious_percentage:.2f}%")
        
        if malicious_percentage < 10:
            st.success("The file is classified as NOT malicious based on the threshold of 10%.")
        else:
            st.error("The file contains 10% or more malicious records and is classified as malicious.")
