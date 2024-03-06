import pandas as pd
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 2:4].values
    Y = data.iloc[:, :1].values
    return X, Y

def main():
    data = pd.read_csv('T3resin1.txt')

    random_indices = np.random.permutation(len(data))
    num_training_samples = int(len(data) * 0.7)
    num_validation_samples = int(len(data) * 0.15)
    num_test_samples = int(len(data) * 0.15)

    training_data = data.iloc[random_indices[:num_training_samples]]
    training_data.to_csv('training_data.csv', index=False)

    test_data = data.iloc[random_indices[-num_test_samples:]]
    test_data.to_csv('test_data.csv', index=False)

if __name__ == "__main__":
    main()
