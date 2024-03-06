import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def visualize_training_data(X_train, Y_train):
    X_class0 = np.asmatrix([X_train[i] for i in range(len(X_train)) if Y_train[i] == 0])
    X_class1 = np.asmatrix([X_train[i] for i in range(len(X_train)) if Y_train[i] == 1])
    plt.scatter([X_class0[:, 0]], [X_class0[:, 1]], color='red')
    plt.scatter([X_class1[:, 0]], [X_class1[:, 1]], color='green')
    plt.xlabel('Total Serum Thyroxin')
    plt.ylabel('Total Serum Triiodothyronine')
    plt.legend(['Having Thyroid', 'Normal'])
    plt.title('Visualization of training data')

def visualize_predictions(X_train, Y_train, query_point, model):
    X_class0 = np.asmatrix([X_train[i] for i in range(len(X_train)) if Y_train[i] == 0])
    X_class1 = np.asmatrix([X_train[i] for i in range(len(X_train)) if Y_train[i] == 1])
    plt.scatter([X_class0[:, 0]], [X_class0[:, 1]], color='red')
    plt.scatter([X_class1[:, 0]], [X_class1[:, 1]], color='green')
    plt.scatter(query_point[0], query_point[1], marker='^', s=75, color='black')
    plt.xlabel('Total Serum Thyroxin')
    plt.ylabel('Total Serum Triiodothyronine')
    plt.legend(['Having Thyroid', 'Normal'])
    plt.title('Prediction using KNN')

    neighbors_object = NearestNeighbors(n_neighbors=5)
    neighbors_object.fit(X_train)
    distances_of_nearest_neighbors, indices_of_nearest_neighbors_of_query_point = neighbors_object.kneighbors(
        [query_point])
    nearest_neighbors_of_query_point = X_train[indices_of_nearest_neighbors_of_query_point[0]]
    plt.scatter(nearest_neighbors_of_query_point[:, 0], nearest_neighbors_of_query_point[:, 1], marker='s', s=60,
                color='blue', alpha=0.30)

def evaluate_model(model, X_val, Y_val, dataset_name):
    validation_set_predictions = [model.predict(X_val[i].reshape((1, 2)))[0] for i in range(X_val.shape[0])]
    validation_misclassification_percentage = sum(1 for i in range(len(validation_set_predictions)) if
                                                   validation_set_predictions[i] != Y_val[i]) * 100 / len(Y_val)
    print(f'Validation percentage on {dataset_name} = {validation_misclassification_percentage:.2f}%')

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 2:4].values
    Y = data.iloc[:, :1].values
    return X, Y

def main():

    X_train, Y_train = load_data('training_data.csv')

    X_train[:, 0] *= 0.1  
    X_train[:, 1] *= 10   

    model = KNeighborsClassifier(n_neighbors=5)  
    model.fit(X_train, Y_train.ravel())

    query_point_training = np.array([1.2, 30])    
    query_point_prediction = np.array([1.2, 30])  

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    visualize_training_data(X_train, Y_train)
    plt.scatter(query_point_training[0], query_point_training[1], marker='o', s=75, color='blue')
    plt.title('Visualization of training data')

    plt.subplot(1, 2, 2)
    visualize_predictions(X_train, Y_train, query_point_prediction, model)
    plt.scatter(query_point_prediction[0], query_point_prediction[1], marker='^', s=75, color='black')
    plt.title('Working of the K-NN classification algorithm')

    X_test, Y_test = load_data('test_data.csv')

    evaluate_model(model, X_test, Y_test, "test data")

    plt.show()

if __name__ == "__main__":
    main()