import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import joblib
import pickle


def train_and_save_model(data_path):
    
    df = pd.read_csv(data_path)

    
    df.drop(['source_ip', 'destination_ip', 'Index'], axis=1, inplace=True)

    
    le = LabelEncoder()
    df['protocol'] = le.fit_transform(df['protocol'])

   
    protocol_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    with open("protocol_mapping.pkl", "wb") as f:
        pickle.dump(protocol_mapping, f)
    
    df['Traffic_Label'] = df['Traffic_Label'].map({'Normal Traffic': 1, 'DDoS Traffic': 2})

    
    scaler = MinMaxScaler()
    features_to_scale = ['flow_duration', 'mean_forward_iat', 'min_forward_iat',
                         'max_forward_iat', 'std_forward_iat', 'mean_backward_iat',
                         'min_backward_iat', 'max_backward_iat', 'std_backward_iat',
                         'mean_flow_iat', 'min_flow_iat', 'max_flow_iat', 'std_flow_iat',
                         'mean_active_time', 'min_active_time', 'max_active_time', 
                         'std_active_time', 'mean_idle_time', 'min_idle_time', 
                         'max_idle_time', 'std_idle_time']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    
    X = df.drop('Traffic_Label', axis=1)
    y = df['Traffic_Label']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr=LogisticRegression()
    dtc=DecisionTreeClassifier()

   
    dtc.fit(X_train, y_train)

    with open("main_model.pkl", "wb") as f:
        pickle.dump(dtc, f)

    
    y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    train_and_save_model("Training _Data.csv")