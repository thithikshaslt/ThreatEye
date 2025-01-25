import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder

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

# for file in sample_files:
#     st.download_button(
#         label=f"Download {file['name']}",
#         data=file['url'],
#         file_name=f"{file['name']}.csv",
#         mime="text/csv"
#     )

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
       'Idle Min']

    st.header("Required Columns")
    st.write("The following columns are required for the analysis:")
    st.write(required_columns)

    missing = []
    for i in list(data.columns):
        if i not in required_columns:
            missing.append(i)
    for i in missing:
        data = data.drop(i, axis=1)

    if len(list(data.columns)) < len(required_columns):
        missing_cols = list(set(required_columns) - set(list(data.columns)) ) 
        st.write("Dataset is not complete. Please upload the complete dataset. The dataset needs the following columns :" + str(missing_cols))
    elif list(data.columns).sort() != required_columns.sort():
        st.write("Column(s) mismatch")
    else:
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        st.header("Data Preprocessing")
        
        st.write("Handling missing values...")

        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        non_numeric_cols = data.select_dtypes(exclude=['float64', 'int64']).columns

        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        for col in non_numeric_cols:
            data[col] = data[col].fillna(data[col].mode()[0])

        # label_encoder = LabelEncoder()
        # data['Label'] = label_encoder.fit_transform(data['Label'])
        # st.write("Encoded Label Column:")

        constant_cols = data.columns[data.nunique() <= 1].tolist()

        numeric_cols = data.select_dtypes(include=[np.number])
        numeric_cols.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = numeric_cols.fillna(numeric_cols.mean())
        data[numeric_cols.columns] = numeric_cols

        st.write("Preprocessed Data Preview:")
        st.dataframe(data.head())

        st.header("Feature Scaling")
        st.dataframe(data.head()) 

        X = data
        if 'Label' in data.columns:
            X = data.drop('Label', axis=1)
            # y = data['Label']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = X_scaled

        st.write("Done feature scaling.")
        st.dataframe(data.head()) 

        st.header("Querying the models")

        models = [
            "Logistic Regression", 
            "Decision Tree", 
            "Random Forest",
            "KNN" 
        ]

        for model_name in models:
            st.subheader(f"{model_name}")

            name = f"models/{model_name.replace(' ', '_')}_model.pkl"
            print(name)
            with open(name, 'rb') as f:
                model = pickle.load(f)
            print(model)

            st.write(f"Loaded Model: {type(model).__name__}")
            y_pred = model.predict(X)
            y_pred_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

            vectorized = np.vectorize(lambda x: 'Anomaly' if x == 1 else 'Normal')

            labels = vectorized(y_pred)
            st.write(labels)
            data[f'{model_name}_Prediction'] = y_pred
            data[f'{model_name}_Label'] = data[f'{model_name}_Prediction'].apply(lambda x: 'Anomaly' if x == 1 else 'Normal')

        st.header("Model Metrics Summary")
        # metrics_df = pd.DataFrame(model_metrics)
        # st.dataframe(metrics_df)

        st.subheader("Final Predicted Labels for the Entire Dataset")
        # model_columns = required_columns.copy()
        model_columns = [f'{model}_Label' for model in models]
        # model_columns.append("Label")
        data['Majority prediction'] = data[model_columns].mode(axis=1)[0]
        model_columns.append('Majority prediction')
        st.write(data[model_columns])
        st.write(data['Majority prediction'])

        st.header("Final Output")

        models = [f'{model}_Label' for model in models]

        # data['Any_Model_Malicious'] = data[models].apply(lambda x: 'Anomaly' in x.values, axis=1)

        # malicious_counts = data['Any_Model_Malicious'].sum()
        malicious_counts = (data['Majority prediction'] == 'Anomaly').sum()

        total_records = data.shape[0]

        malicious_percentage = (malicious_counts / total_records) * 100 if total_records > 0 else 0

        st.write(f"Total Records: {total_records}")
        st.write(f"Malicious Records: {malicious_counts}")
        st.write(f"Percentage of Malicious Records: {malicious_percentage:.2f}%")

    # if malicious_percentage < 10:
    #     st.success("The file is classified as NOT malicious based on the threshold of 10%.")
    # else:
    #     st.error("The file contains 10% or more malicious records and is classified as malicious.")
else:
    st.write("Please upload a CSV file to begin.")



