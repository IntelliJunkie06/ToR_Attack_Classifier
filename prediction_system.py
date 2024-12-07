import pandas as pd
import joblib
import numpy as np

model_path = "main_model.pkl"
scaler_path = "scaler.pkl"
protocol_mapping_path = "protocol_mapping.pkl"
input_csv = "test.csv"
output_csv = "new_output.csv"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
protocol_mapping = joblib.load(protocol_mapping_path)

data = pd.read_csv(input_csv)

columns_to_drop = ['source_ip', 'destination_ip']
data.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')


if "protocol" in data.columns:
    data["protocol"] = data["protocol"].map(protocol_mapping)
    data.to_csv("check.csv")
    print("First Checkpoint")
    if data["protocol"].isnull().any():
        raise ValueError("Some protocol values in the input CSV could not be mapped.")


required_features = [
    'protocol', 'flow_duration', 'mean_forward_iat', 'min_forward_iat', 'max_forward_iat', 'std_forward_iat',
    'mean_backward_iat', 'min_backward_iat', 'max_backward_iat', 'std_backward_iat', 'mean_flow_iat',
    'min_flow_iat', 'max_flow_iat', 'std_flow_iat', 'mean_active_time', 'min_active_time',
    'max_active_time', 'std_active_time', 'mean_idle_time', 'min_idle_time', 'max_idle_time', 'std_idle_time'
]

scaled_features = [
    'flow_duration', 'mean_forward_iat', 'min_forward_iat', 'max_forward_iat', 'std_forward_iat',
    'mean_backward_iat', 'min_backward_iat', 'max_backward_iat', 'std_backward_iat', 'mean_flow_iat',
    'min_flow_iat', 'max_flow_iat', 'std_flow_iat', 'mean_active_time', 'min_active_time',
    'max_active_time', 'std_active_time', 'mean_idle_time', 'min_idle_time', 'max_idle_time', 'std_idle_time'
]

missing_features = [feature for feature in required_features if feature not in data.columns]
if missing_features:
    raise ValueError(f"Input CSV is missing required features: {missing_features}")


X_scaled = data[scaled_features]
X_scaled = scaler.fit_transform(X_scaled)
X_scaled_df = pd.DataFrame(X_scaled, columns=scaled_features)
X_scaled_df.to_csv("Check2.csv")
X_combined = pd.concat([data[['protocol']].reset_index(drop=True),X_scaled_df], axis=1) 
X_combined_df=pd.DataFrame(X_combined)
X_combined_df.to_csv("Check3.csv")

data['Prediction'] = model.predict(X_combined_df)

prediction_mapping = {1: "Normal Traffic", 2: "DDoS Traffic"}
data['Prediction'] = data['Prediction'].map(prediction_mapping)


data.to_csv(output_csv, index=False)

print(f"Prediction completed. Results saved to {output_csv}")
