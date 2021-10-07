import numpy as np, pickle, gzip
from numpy.random import uniform as runi

class Utils:
    def readData(path):
        f = gzip.open(path, 'rb')
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        f.close()

        return train_set, valid_set, test_set

class DigitNetwork:
    def __init__(self, inputs=None, labels=None, noParameters=False):

        if not noParameters:
            self.inputs = inputs
            self.nrInputs = len(self.inputs)
            self.inputSize = len(inputs[0])
            self.labels = labels
            self.nrPerceptrons = 10
            self.nrDigits = 10

            self.weights = runi(0, 1, (self.nrPerceptrons, self.inputSize))
            self.biases = np.random.uniform(0, 1, self.nrPerceptrons)
            self.targets = np.zeros((self.nrDigits, self.nrDigits))
            np.fill_diagonal(self.targets, 1)
    
    def activate(self, input):
        if input > 0:
            return 1
        return 0

    def train(self, nrIterations, learnRate):
        permutation = np.arange(self.nrInputs)

        for iteration in range(nrIterations):
            np.random.shuffle(permutation)
            print(f"Iteration {iteration+1}")

            for index in permutation:
                netInput = np.dot(self.weights, self.inputs[index]) + self.biases
                output = np.array(
                    [self.activate(netInput[i]) for i in range(self.nrPerceptrons)]
                )
                toDifference = (self.targets[self.labels[index]] - output) * learnRate
                self.biases += toDifference

                for i in range(self.nrPerceptrons):
                    self.weights[i] += self.inputs[index] * toDifference[i]

    def classify(self, input):
        return np.argmax(np.dot(self.weights, input) + self.biases)

    def test(self, inputs, labels):
        correctlyClassified = 0

        for i in range(len(inputs)):
            if self.classify(inputs[i]) == labels[i]:
                correctlyClassified += 1
        
        print(f"Correctly classified {100.0 * correctlyClassified / len(inputs)}%")

    def save(self):
        with open('./weights.npy', 'wb') as f:
            np.save(f, self.weights)
        
        with open('./biases.npy', 'wb') as f:
            np.save(f, self.biases)
    
    def load(self):
        with open('./weights.npy', 'rb') as f:
            self.weights = np.load(f)
        
        with open('./biases.npy', 'rb') as f:
            self.biases = np.load(f)

if __name__ == '__main__':
    train_set, valid_set, test_set = Utils.readData('./mnist.pkl.gz')

    network = DigitNetwork(train_set[0], train_set[1])
    network.train(3, 0.02)
    network.test(test_set[0], test_set[1])