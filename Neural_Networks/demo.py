import numpy as np

class NeuralNetwork():
	
	# Initialize the weights and set the random seed
	def __init__(self):

		#Seed the random numbber generator , so it generates the same number
		#everytime the program runs
		np.random.seed(1)

		# We model a single neuron, with 3 input connections and 1 output connection
		# we assign random weights to a 3x1 matrix, with values in range -1 to 1
		# with mean 0
		self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

	# The sigmoid function, which describes an s shaped curve
	# we pass the weighted sum of the inuts through this function
	# to normalise them b/w 0 and 1
	def __sigmoid(self, x):
		return 1/(1 + np.exp(-x))

	#gradient of the sigmoid
	def __sigmoid_derivative(self, x):
		return x * (1-x)

	def train(self, training_set_inputs, training_set_outputs, iterations):
		for ix in xrange(iterations):
			#pass the training set through our neural network

			output = self.predict(training_set_inputs)

			#calculate the error
			error = training_set_outputs - output

			#multiply the error by the input and again by the gradient of the sigmoid curve
			adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

			#ADJUST THE weights
			self.synaptic_weights += adjustment


	def predict(self, inputs):
		# pass inputs through our neural network(our  single neuron)
		return self.__sigmoid(np.dot(inputs, self.synaptic_weights))


if __name__ == '__main__':

	#Inititalise a single neuron neural network
	neural_network = NeuralNetwork()

	print 'Random starting synaptic weights'
	print neural_network.synaptic_weights

	#The training set. We have 4 examples, each consisting of 3 input values
	#and 1 output value.

	training_set_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0, 1, 1]])
	training_set_outputs =  np.array([0, 1, 1, 0]).T

	#train the neural network using a training set.
	#Do it 10, 000 times and make small adjustments each time
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print 'New synaptic weights after training: '
	print neural_network.synaptic_weights

	#Test the neural network
	print 'predicting:'
	print neural_network.predict(np.array([1, 0, 0]))
