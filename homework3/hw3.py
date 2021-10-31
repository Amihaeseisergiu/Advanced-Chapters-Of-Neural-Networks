import numpy as np, pickle, gzip, os
from numpy.random import normal as rnorm, uniform as runif, shuffle, binomial
from scipy import ndimage
from scipy.ndimage import rotate, shift

class Utility:
    def read_data(path):
        f = gzip.open(path, 'rb')
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        f.close()

        return train_set, valid_set, test_set
    
    def one_hot(data, size):
        one_hot_vector = np.zeros((size, 1))
        one_hot_vector[data] = 1
        
        return one_hot_vector

    def column_vector(data):
        column_vectors = []

        for vector in data:
            column_vectors.append(vector.reshape(len(vector), 1))
        
        return column_vectors

    def increase_dataset(inputs, labels, output):
        new_inputs = []
        new_labels = []
        print("Increasing dataset")

        for i in range(len(inputs)):
            size = int(np.sqrt(len(inputs[i])))

            reshaped = inputs[i].reshape(size, size)
            angle = runif(-25, 25)
            rotated = rotate(reshaped, angle, reshape=False)
            shifted = shift(rotated, (runif(-3, 3), runif(-3, 3)))

            new_inputs.append(rotated.reshape(len(inputs[i]), 1))
            new_labels.append(Utility.one_hot(labels[i], output))
            new_inputs.append(shifted.reshape(len(inputs[i]), 1))
            new_labels.append(Utility.one_hot(labels[i], output))
            new_inputs.append(inputs[i].reshape(len(inputs[i]), 1))
            new_labels.append(Utility.one_hot(labels[i], output))

            if (i + 1) % (len(inputs) / 4) == 0:
                    print(f"{int(1 / (len(inputs) / (i + 1)) * 100)}% complete")
        
        return new_inputs, new_labels

class Function:
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def d_sigmoid(y):
        return y * (1 - y)
    
    def softmax(z):
        numerator = np.exp(z)
        return numerator / np.sum(numerator)

    def d_cost(y, t):
        return y - t

class DigitNetwork:
    def __init__(self, input=None, output=None, *hidden, load=False):

        if not load:
            new_inputs, new_labels = Utility.increase_dataset(input[0], input[1], output)

            self.inputs = new_inputs
            self.labels = new_labels
            self.n_inputs = len(self.inputs)
            self.input_size = len(self.inputs[0])
            self.hidden_size = len(hidden)

            self.weights = []
            self.biases = []
            last_layer_size = self.input_size

            for hidden_size in hidden:
                self.weights.append(rnorm(0, 1.0 / np.sqrt(last_layer_size),
                (hidden_size, last_layer_size)))
                self.biases.append(rnorm(0, 1.0, (hidden_size, 1)))
                last_layer_size = hidden_size
            
            self.weights.append(rnorm(0, 1.0 / np.sqrt(last_layer_size),
                (output, last_layer_size)))
            self.biases.append(rnorm(0, 1.0, (output, 1)))

    def feed_foward(self, input, dropout_percent):
        activations = [input]

        for i in range(len(self.weights)):
            if i + 1 == len(self.weights):
                input = Function.softmax(np.dot(self.weights[i], input) + self.biases[i])
            else:
                input = Function.sigmoid(np.dot(self.weights[i], input) + self.biases[i])

                if dropout_percent != -1:
                    hidden_neurons = binomial(1, 1 - dropout_percent, (len(input), 1))
                    input = np.multiply(input, hidden_neurons) * (1 / (1 - dropout_percent))

            activations.append(input)

        return activations

    def backpropagate(self, output, activations):
        weights = [np.zeros(weight.shape) for weight in self.weights]
        biases = [np.zeros(bias.shape) for bias in self.biases]

        error = Function.d_cost(activations[-1], output)

        weights[-1] = np.dot(error, activations[-2].T)
        biases[-1] = error

        for i in range(2, len(self.weights) + 1):
            error = np.dot(self.weights[-i + 1].T, error) * Function.d_sigmoid(activations[-i])

            weights[-i] = np.dot(error, activations[-i-1].T)
            biases[-i] = error

        return weights, biases

    def train(self, n_iterations, batch_size, learning_rate, dropout_percent, maxnorm):
        permutation = np.arange(self.n_inputs)
        n_batches = self.n_inputs // batch_size
        batch_learning_rate = learning_rate / batch_size

        print(f"Started training with {n_batches} batches")

        for iteration in range(n_iterations):
            print(f"[!] Iteration {iteration + 1} begins")
            shuffle(permutation)

            for i in range(n_batches):
                batch_weights = [np.zeros(weight.shape) for weight in self.weights]
                batch_biases = [np.zeros(bias.shape) for bias in self.biases]

                for j in range(batch_size):
                    activations = self.feed_foward(
                        self.inputs[permutation[i*batch_size + j]], dropout_percent)
                    weights, biases = self.backpropagate(
                        self.labels[permutation[i*batch_size + j]], activations)
                    
                    for k in range(len(weights)):
                        batch_weights[k] += weights[k]
                        batch_biases[k] += biases[k]
                    
                batch_learning_rate = learning_rate / batch_size

                for k in range(len(batch_weights)):
                    self.weights[k] -= batch_learning_rate * batch_weights[k]
                    self.biases[k] -= batch_learning_rate * batch_biases[k]

                    weights_norm = np.sum(self.weights[k] ** 2, axis=0)
                    for l in range(len(self.weights[k][0])):
                        if weights_norm[l] > maxnorm:
                            self.weights[k][:,l] *= maxnorm / (weights_norm[l] + 10 ** -8)
                
                if (i + 1) % (n_batches / 4) == 0:
                    print(f"[{iteration + 1}] Processed {int(1 / (n_batches / (i + 1)) * 100)}% of the batch")

    def classify(self, input):
        return np.argmax(self.feed_foward(input, -1)[-1])

    def test(self, inputs, labels):
        column_inputs = Utility.column_vector(inputs)

        correctly_classified = sum(
            1 if self.classify(column_inputs[i]) == labels[i] else 0
            for i in range(len(column_inputs))
        )

        print(f"Correctly classified {100 * correctly_classified / len(inputs)}%")

    def save(self):
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, r'network_data')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)

        for i in range(len(self.weights)):
            with open(f'./network_data/weights_{i}.npy', 'wb') as f:
                np.save(f, self.weights[i])
        
            with open(f'./network_data/biases_{i}.npy', 'wb') as f:
                np.save(f, self.biases[i])
    
    def load(self):
        self.weights = []
        self.biases = []

        script_dir = os.path.dirname(__file__)
        network_path = os.path.join(script_dir, 'network_data')
        n_weights = len([name for name in os.listdir(network_path)
         if os.path.isfile(os.path.join(network_path, name))]) // 2

        for i in range(n_weights):
            weights_path = os.path.join(network_path, f'weights_{i}.npy')
            biases_path = os.path.join(network_path, f'biases_{i}.npy')

            with open(weights_path, 'rb') as f:
                self.weights.append(np.load(f))

            with open(biases_path, 'rb') as f:
                self.biases.append(np.load(f))

if __name__ == '__main__':
    train, validation, test = Utility.read_data("./digits_data/mnist.pkl.gz")
    dn = DigitNetwork(train, 10, 100)
    dn.train(2, 10, 0.2, 0.1, 5)
    dn.test(test[0], test[1])
    #dn.save()