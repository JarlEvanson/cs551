import os
import subprocess
import tempfile
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

import streamlit

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        # The Encoder: This is what we keep for the final IDS
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        # Projection Head: Used only during SSL pre-training
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 32)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        projection = self.projector(embedding)
        return embedding, projection

label_encoder = joblib.load('label_encoder.joblib')
preprocessor = joblib.load('preprocessor.joblib')

selected_features = joblib.load('selected_features.joblib')

base_lr = joblib.load('base_linear_regression.joblib')
ssl_lr = joblib.load('ssl_linear_regression.joblib')
base_rf = joblib.load('base_random_forest.joblib')
ssl_rf = joblib.load('ssl_random_forest.joblib')

# Initialize and load weights.
input_dim = preprocessor.transformers_[0][2].shape[0]

full_model = ContrastiveModel(input_dim=input_dim)
full_model.load_state_dict(torch.load('ssl_encoder.pth'))
full_model.eval()

# Only use the encoder part.
ssl_encoder = full_model.encoder

streamlit.title('Network Intrusion Dection System (CS 551 Final Project)')

pcap_file = streamlit.file_uploader(
    'PCAP file',
    type = 'pcap'
)

if not pcap_file:
    streamlit.stop()

with tempfile.TemporaryDirectory() as tmpdirname:
    config_path = os.path.join(tmpdirname, "config.json")
    pcap_path = os.path.join(tmpdirname, "file.pcap")
    csv_path = os.path.join(tmpdirname, "file.csv")
    
    with open(pcap_path, "wb") as f:
        f.write(pcap_file.getbuffer())

    with open(config_path, "w") as f:
        config = {
            'pcap_file_address': pcap_path,
            'output_file_address': csv_path,
        }
        json.dump(config, f, indent=4)

    streamlit.subheader("Pre-Processing Logs")

    log_placeholder = streamlit.empty()
    log_text = ""

    subprocess = subprocess.Popen(
        ['ntlflowlyzer', '--config', config_path],
        stdout=subprocess.PIPE, 
        text=True
    )

    for line in subprocess.stdout:
        log_text += f"{line.strip()}\n"
        log_placeholder.code(log_text)

    subprocess.wait()

    csv_path = os.path.join(tmpdirname, "file.csv")
    if not os.path.exists(csv_path):
        streamlit.stop()

    # Read the processed CSV.
    df = pd.read_csv(csv_path).copy()

    # Compute the engineered features.
    df['fwd_header_payload_ratio'] = np.divide(
        df['fwd_total_header_bytes'].to_numpy(),
        df['fwd_total_payload_bytes'].to_numpy(),
        out=np.zeros_like(df['fwd_total_payload_bytes'].to_numpy(), dtype=float),
        where=df['fwd_total_payload_bytes'].to_numpy() != 0
    )

    df['bwd_header_payload_ratio'] = np.divide(
        df['bwd_total_header_bytes'].to_numpy(),
        df['bwd_total_payload_bytes'].to_numpy(),
        out=np.zeros_like(df['bwd_total_payload_bytes'].to_numpy(), dtype=float),
        where=df['bwd_total_payload_bytes'].to_numpy() != 0
    )

    df['activity_ratio'] = np.divide(
        df['active_mean'].to_numpy(),
        df['idle_mean'].to_numpy(),
        out=np.zeros(len(df)),
        where=df['idle_mean'].to_numpy() != 0
    )

    df['payload_length_cov'] = np.divide(
        df['payload_bytes_std'].to_numpy(),
        df['payload_bytes_mean'].to_numpy(),
        out=np.zeros(len(df)),
        where=df['payload_bytes_mean'].to_numpy() != 0
    )

    # Drop features that aren't utilized.
    df = df[selected_features]

    # Pre-process dataset.
    df_scaled = preprocessor.transform(df)
    with torch.no_grad():
        df_ssl = ssl_encoder(torch.tensor(df_scaled, dtype=torch.float32)).numpy()

    # Predict their status.
    base_lr_predictions = base_lr.predict(df_scaled)
    ssl_lr_predictions = ssl_lr.predict(df_ssl)
    
    base_rf_predictions = base_rf.predict(df_scaled)
    ssl_rf_predictions = ssl_rf.predict(df_ssl)

    # Convert predictions to their string labels.
    df['Logistic Regression'] = label_encoder.inverse_transform(base_lr_predictions)
    df['Logistic Regression (SSL)'] = label_encoder.inverse_transform(ssl_lr_predictions)
    
    df['Random Forest'] = label_encoder.inverse_transform(base_rf_predictions)
    df['Random Forest (SSL)'] = label_encoder.inverse_transform(ssl_rf_predictions)

    # Rotate the newly added lists to the start.
    cols = df.columns.tolist()
    shifted_cols = cols[-4:] + cols[:-4]
    df = df[shifted_cols]

    # Enable viewing of the individual flows and the predicted values.
    streamlit.header("Flow Predictions & Features")
    streamlit.dataframe(df)

    # Begin the SHAP explainer.
    streamlit.divider()
    streamlit.subheader("Interactive SHAP Explanation")

    col1, col2 = streamlit.columns(2)
    
    with col1:
        # Selection for the Model
        model_choice = streamlit.selectbox(
            "Select Model to Explain:",
            options=[
                "Logistic Regression",
                "Logistic Regression (SSL)",
                "Random Forest",
                "Random Forest (SSL)"
            ]
        )
    
    with col2:
        # Selection for the Flow
        flow_ids = [f"Flow {i}" for i in range(len(df))]
        selected_flow_label = streamlit.selectbox(
            "Select Flow to Explain:",
            options=flow_ids
        )

    # Extract the index from the selection.
    selected_index = flow_ids.index(selected_flow_label)

    if "Random Forest" in model_choice:
        if "SSL" in model_choice:
            current_model = ssl_rf
            # Use the SSL embeddings (which will be named by index).
            feature_names_ssl = [f"Embedding_{i}" for i in range(df_ssl.shape[1])]
            shap_data = pd.DataFrame(df_ssl, columns=feature_names_ssl)
            current_prediction = ssl_rf_predictions[selected_index]
        else:
            current_model = base_rf
            shap_data = pd.DataFrame(df_scaled, columns=preprocessor.get_feature_names_out())
            current_prediction = base_rf_predictions[selected_index]

        explainer = shap.TreeExplainer(current_model)
    else:
        if "SSL" in model_choice:
            current_model = ssl_lr
            # Use the SSL embeddings (which will be named by index).
            feature_names_ssl = [f"Embedding_{i}" for i in range(df_ssl.shape[1])]
            shap_data = pd.DataFrame(df_ssl, columns=feature_names_ssl)
            current_prediction = ssl_lr_predictions[selected_index]
        else:
            current_model = base_lr
            shap_data = pd.DataFrame(df_scaled, columns=preprocessor.get_feature_names_out())
            current_prediction = base_lr_predictions[selected_index]

        explainer = shap.LinearExplainer(current_model, shap_data)

    # Try to explain why the class was predicted.
    shap_explainer = explainer(shap_data.iloc[selected_index: selected_index + 1])

    fig, ax = plt.subplots(figsize=(10, 6))
    streamlit.write(f"Explaining **{model_choice}** for **{selected_flow_label}**")

    predicted_class_idx = base_rf_predictions[selected_index]
    shap.plots.waterfall(
        shap_explainer[0, :, current_prediction],
        max_display=40,
        show=False
    )

    streamlit.pyplot(plt.gcf())
