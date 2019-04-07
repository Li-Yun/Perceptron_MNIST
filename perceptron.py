import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, in_train, in_test, lr = 0.01, max_epoch = 30):
        self.lr = lr
        self.training_data = in_train
        self.testing_data = in_test
        self.max_epoch = max_epoch
        self.weights = np.random.uniform(-0.05, 0.05, (785, 10))
    def fit(self):
        for epoch in range(1, self.max_epoch + 1):
            print('Epoch: ', epoch)
            # iterate through all the training examples
            for data_index in range(self.training_data.shape[0]):
                # one-hot encoding
                target_vector = np.zeros(10)
                target_vector[int(self.training_data[data_index, 0])] = 1.0

                # compute the output
                y_hat = np.dot(self.training_data[data_index, 1:self.training_data.shape[1]], self.weights)
                y_hat[y_hat > 0] = 1.0
                y_hat[y_hat <= 0] = 0.0

                # update the weights
                self.weights += self.lr * np.outer(self.training_data[data_index, 1:self.training_data.shape[1]], np.subtract(target_vector, y_hat))
    def predict(self):
        output = np.dot(self.testing_data[:, 1:self.testing_data.shape[1]], self.weights)
        correct_num = 0
        
        for index in range(self.testing_data.shape[0]):
            pred = np.argmax(output[index, :])
            if pred == int(self.testing_data[index, 0]):
                correct_num += 1
        return correct_num / float(self.testing_data.shape[0]) * 100.0

def data_loading():
    out_data = {}
    training = pd.read_csv('mnist_train.csv', header = None).values
    training = np.asarray(training, dtype = float)
    training[:, 1:785] = training[:, 1:785] / 255.0
    training = np.append(training, np.ones((training.shape[0], 1)), axis = 1)
    out_data['train_data'] = training

    testing = pd.read_csv('mnist_test.csv', header = None).values
    testing = np.asarray(testing, dtype = float)
    testing[:, 1:785] = testing[:, 1:785] / 255.0
    testing = np.append(testing, np.ones((testing.shape[0], 1)), axis = 1)
    out_data['test_data'] = testing

    return out_data

def main():
    # data loading
    data_collect = data_loading()

    single_network = Perceptron(data_collect['train_data'], data_collect['test_data'])
    # train 10 perceptrons
    single_network.fit()

    # prediction computation
    accuracy = single_network.predict()
    print('Testing Accuracy: ', accuracy)
    
if __name__ == "__main__":
    main()
