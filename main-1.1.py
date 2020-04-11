import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)

        self.learning_rate = 10

        self.synaptic_weights = 2 * np.random.standard_normal((2, 1)) - 1

    def sigmoid(self, x, deriv = False):
        if deriv == True:
            return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
        return 1 / (1 + np.exp(-x))

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)

            error = training_set_outputs - output

            adjustment = np.dot(training_set_inputs.T, error * self.sigmoid(output,deriv=True))

            self.synaptic_weights += adjustment * self.learning_rate

    def think(self, inputs):
        return self.sigmoid(np.dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

    iterations = 10000

    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)

    training_set_inputs = np.array(([[3,1.5],
                                     [2,1],
                                     [4,1.5],
                                     [3,1],
                                     [4,0.5],
                                     [2,0.5],
                                     [5.5,1],
                                     [1,1]]))

    training_set_outputs = np.array([[1],
                                     [0],
                                     [1],
                                     [0],
                                     [1],
                                     [0],
                                     [1],
                                     [0]])

    neural_network.train(training_set_inputs, training_set_outputs, iterations)

    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)

    print ("Considering new situation [4.5, 1] -> ?: ")
    print (neural_network.think(np.array([[4.5, 1]])))
