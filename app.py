from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os
import secrets
import joblib

app = Flask(__name__)

app.secret_key = secrets.token_hex(16)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'  
ALLOWED_EXTENSIONS = {'csv'}


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = "static/main_model.pkl"
scaler_path = "scaler.pkl"
protocol_mapping_path = "static/protocol_mapping.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
protocol_mapping = joblib.load(protocol_mapping_path)


def allowed_file(filename):
    """Check if the uploaded file is allowed (CSV only)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def classifier():
    """Render the classifier page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process the prediction."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Process the file
            output_file = process_file(file_path)
            flash('File successfully processed!')
            return redirect(url_for('download_result', filename=output_file))
        except Exception as e:
            flash(f"Error processing file: {str(e)}")
            return redirect(url_for('classifier'))

    flash('Invalid file type. Only CSV files are allowed.')
    return redirect(url_for('classifier'))


def process_file(file_path):
    """Process the uploaded file and predict traffic."""
    data = pd.read_csv(file_path)

    # Drop unnecessary columns
    columns_to_drop = ['source_ip', 'destination_ip']
    data.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

    # Map protocols
    if "protocol" in data.columns:
        data["protocol"] = data["protocol"].map(protocol_mapping)
        if data["protocol"].isnull().any():
            raise ValueError("Some protocol values in the input CSV could not be mapped.")

    # Ensure required features are present
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

    # Scale the features
    X_scaled = scaler.transform(data[scaled_features])
    X_scaled_df = pd.DataFrame(X_scaled, columns=scaled_features)

    # Combine scaled features and protocol column
    X_combined = pd.concat([data[['protocol']].reset_index(drop=True), X_scaled_df], axis=1)

    # Predict and map the predictions
    data['Prediction'] = model.predict(X_combined)
    prediction_mapping = {1: "Normal Traffic", 2: "DDoS Traffic"}
    data['Prediction'] = data['Prediction'].map(prediction_mapping)

    # Save the results
    output_file = os.path.join(RESULT_FOLDER, 'result.csv')
    data.to_csv(output_file, index=False)

    return 'result.csv'


@app.route('/download/<filename>')
def download_result(filename):
    """Provide a link to download the processed result file."""
    file_path = os.path.join(RESULT_FOLDER, filename)

    # Check if the file exists
    if os.path.exists(file_path):
        return render_template('download.html', filename=filename)
    else:
        return f'''
            <h1>Error: File not found</h1>
            <p>Sorry, the requested file could not be found.</p>
        '''


@app.route('/training')
def training():
    return render_template('training.html')


@app.route('/methodology')
def methodology():
    return render_template('methodology.html')


if __name__ == '__main__':
    app.run(debug=True)
