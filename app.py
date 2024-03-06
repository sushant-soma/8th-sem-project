from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__, static_url_path='/static')

# Load the dataset
data = pd.read_csv('T3resin1.txt')

# Preprocess the data
X = data.iloc[:, 2:4].values
Y = data.iloc[:, :1].values

# Split the data into training and test sets
random_indices = np.random.permutation(len(Y))
num_training_samples = int(len(Y) * 0.7)

x_train = X[random_indices[:num_training_samples]]
y_train = Y[random_indices[:num_training_samples]]

x_test = X[random_indices[num_training_samples:]]
y_test = Y[random_indices[num_training_samples:]]

# Train the logistic regression model
model = LogisticRegression(C=1e5)
model.fit(x_train, y_train)

# Route to get dataset
@app.route('/get_dataset')
def get_dataset():
    dataset_type = request.args.get('dataset')
    if dataset_type == 'test_data':
        dataset = pd.read_csv('test_data.csv').values.tolist()
    elif dataset_type == 'training_data':
        dataset = pd.read_csv('training_data.csv').values.tolist()
    else:
        dataset = data.values.tolist()
    return jsonify(dataset)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/detect_cancer', methods=['POST'])
def detect_cancer():
    if request.method == 'POST':
        total_serum_thyroxin = float(request.form['total_serum_thyroxin'])
        total_serum_triiodothyronine = float(request.form['total_serum_triiodothyronine'])
        
        # Predict whether thyroid cancer is present or not
        prediction = model.predict([[total_serum_thyroxin, total_serum_triiodothyronine]])
        result = "Thyroid Cancer Detected" if prediction[0] == 1 else "No Thyroid Cancer Detected"
        
        return jsonify({'result': result})

@app.route('/validate')
def validate():
    # Load test data
    test_data = pd.read_csv('test_data.csv')
    x_test = test_data.iloc[:, 2:4].values
    y_test = test_data.iloc[:, :1].values
    
    # Validate model on test data
    accuracy = model.score(x_test, y_test)
    
    return f'Accuracy on test data: {accuracy:.2f}'

if __name__ == '__main__':
    app.run(debug=True)
