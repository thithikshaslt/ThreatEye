import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, classification_report, roc_curve, auc)

st.title("Intrusion Detection System using Machine Learning")

st.header("1. Load Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = data.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'])
    
    required_columns = ['Src Port', 'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts',
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
       'Idle Min', 'Label']

    if len(data.columns) != len(required_columns):
        st.write("Dataset is not complete. Please upload the complete dataset. The dataset needs the following columns :" + str(required_columns))
    elif data.columns.sort() != required_columns.sort():
        st.write("Column(s) mismatch")
    else:
        st.write("Dataset Preview:")
        st.dataframe(data.head())

    if 'Label' not in data.columns:
        st.warning("Label column not found in the dataset. Please check your dataset for a column containing the target labels.")
        st.write("Columns in dataset:", data.columns)
    else:
        st.header("2. Data Preprocessing")
        
        st.write("Handling missing values...")

        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        non_numeric_cols = data.select_dtypes(exclude=['float64', 'int64']).columns

        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        for col in non_numeric_cols:
            data[col] = data[col].fillna(data[col].mode()[0])

        label_encoder = LabelEncoder()
        data['Label'] = label_encoder.fit_transform(data['Label'])
        st.write("Encoded Label Column:")

        constant_cols = data.columns[data.nunique() <= 1].tolist()
        if 'Label' in list(constant_cols):
            constant_cols.remove('Label')
        data = data.drop(columns=constant_cols)

        numeric_cols = data.select_dtypes(include=[np.number])
        numeric_cols.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = numeric_cols.fillna(numeric_cols.mean())
        data[numeric_cols.columns] = numeric_cols

        st.write("Preprocessed Data Preview:")
        st.dataframe(data.head())

        st.header("3. Statistical Inference")
        
        st.write("Descriptive Statistics:")
        st.write(data.describe())
        
        st.write("Correlation Matrix:")
        corr_matrix = data.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=False, cmap='Blues')
        st.pyplot()

        st.header("4. Feature Scaling")
        st.dataframe(data.head()) 
        X = data.drop('Label', axis=1)
        y = data['Label']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.write("Splitting the data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        class_counts = y_train.value_counts()
        st.write("Class distribution in training data:")
        st.write(class_counts)
        sns.countplot(x=y_train)
        st.pyplot()

        st.header("5. Model Evaluation")

        models = {
            "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            "KNN": KNeighborsClassifier()
        }

        model_metrics = []
        kf = StratifiedKFold(n_splits=5)

        for model_name, model in models.items():
            st.subheader(f"{model_name}")

            # K-fold cross-validation
            name = f"models/{model_name.replace(' ', '_')}_model.pkl"
            print(name)
            with open(name, 'rb') as f:
                model = pickle.load(f)
            print(model)

            # cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
            # st.write(f"K-Fold Cross-Validation Accuracy Scores: {cv_scores}")
            # st.write(f"Mean Accuracy: {np.mean(cv_scores):.5f}")
            
            # Fit the model and evaluate on the test set
            # model.fit(X_train, y_train)
            st.write(f"Loaded Model: {type(model).__name__}")
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            st.write(f"Accuracy on Test Data: {accuracy:.5f}")
            st.write(f"Precision: {precision:.5f}")
            st.write(f"Recall: {recall:.5f}")
            st.write(f"F1-Score: {f1:.5f}")
            
            if y_pred_prob is not None:
                auc_roc = roc_auc_score(y_test, y_pred_prob)
                st.write(f"AUC-ROC: {auc_roc:.5f}")
                
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                plt.figure()
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_roc:.5f})")
                plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve for {model_name}")
                plt.legend()
                st.pyplot()

            conf_matrix = confusion_matrix(y_test, y_pred)
            st.write("Confusion Matrix:")
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            st.pyplot()

            model_metrics.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "AUC-ROC": auc_roc if y_pred_prob is not None else "N/A"
            })

            data[f'{model_name}_Prediction'] = model.predict(X_scaled)
            data[f'{model_name}_Label'] = data[f'{model_name}_Prediction'].apply(lambda x: 'Anomaly' if x == 1 else 'Normal')

        st.header("Model Metrics Summary")
        metrics_df = pd.DataFrame(model_metrics)
        st.dataframe(metrics_df)

        st.subheader("Final Predicted Labels for the Entire Dataset")
        model_columns = [f'{model}_Label' for model in models.keys()]
        st.write(data[model_columns])

        st.header("6. Final Output")

        models = [f'{model}_Label' for model in models.keys()]

        data['Any_Model_Malicious'] = data[models].apply(lambda x: 'Anomaly' in x.values, axis=1)

        malicious_counts = data['Any_Model_Malicious'].sum()

        total_records = data.shape[0]

        malicious_percentage = (malicious_counts / total_records) * 100 if total_records > 0 else 0

        st.write(f"Total Records: {total_records}")
        st.write(f"Malicious Records: {malicious_counts}")
        st.write(f"Percentage of Malicious Records: {malicious_percentage:.2f}%")

        if malicious_percentage < 10:
            st.success("The file is classified as NOT malicious based on the threshold of 10%.")
        else:
            st.error("The file contains 10% or more malicious records and is classified as malicious.")
else:
    st.write("Please upload a CSV file to begin.")



