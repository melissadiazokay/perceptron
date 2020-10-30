import random
import math
from mnist import MNIST

class Neuron(): 

	def __init__(self,activation,bias):

		self.activation = activation
		self.bias = bias

	def getActivation(self):
		return self.activation

	def getBias(self):
		return self.bias

	def setActivation(self,activation):
		self.activation = activation

	def think(self,weights,activations,activationFunction):

		# Update own activation as per Sigma( sum( weights_previous_layer * activations_previous layer) + bias )
		# ---
		# weights (list) - all of the weights from previous layer
		# activations (list) - all of the activations from previous layer
		# activationFunction (function) - activation function from Network
		# * length of weights and activations must be the same

		sigma = 0
		for i in range(len(activations)):
			sigma += weights[i] * activations[i]

		self.activation = activationFunction(sigma + self.bias)


class Network(Neuron): 

	def __init__(self,structure,activationFunction): 

		self.structure = structure
		self.activationFunction = activationFunction
		self.layers = []
		self.weights = []
		self.input = []
		self.output = []

	def initializeNetwork(self):

		# create a new network with random weights and biases

		# all the input weights to all neurons indexed by [layer index][neuron index]
		self.weights = [] * len(self.structure)

		# initialize the weights and network structure 
		for i, n_neurons in enumerate(self.structure):
			layer = []
			self.weights.append([])
			for j in range(n_neurons):
				self.initializeWeights(i)
				bias = random.uniform(-1,1)*10
				activation = random.random()
				layer.append(Neuron(activation,bias))

			self.layers.append(layer)

	def initializeWeights(self,layerIndex):

		# initialize the weights object to random values

		if(layerIndex == 0): # first layer does not have input weights - fill with placeholders
			self.weights[0].append([])
		else: # fill with input weights corresponding to neurons from last layer
			arr = []
			for k in range(self.structure[layerIndex - 1]):
				arr.append(random.uniform(-1,1))
			self.weights[layerIndex].append(arr)

	def runNetwork(self, input = []):

		# run the network
		#
		# input (list) (optional) - a list of input layer activations

		if(len(input) > 0): 
			assert len(input) == len(self.layers[0])
			self.input = input
			for i in enumerate(self.layers[0]):
				self.layers[0][i[0]].setActivation(self.input[i[0]])

		self.feedForward()

	def feedForward(self):

		for i,layer in enumerate(self.layers):
			if(i == 0): continue # skipping it cuz that's the input layer
			activations = self.getLayerActivations(i-1)
			for j,neuron in enumerate(layer):
				weights = self.weights[i][j]
				neuron.think(weights,activations,self.activationFunction)

		self.output = self.getLayerActivations(len(self.layers) - 1)
			
	def getLayerActivations(self,layerIndex):

		activations = []
		for neuron in self.layers[layerIndex]:
			activations.append(neuron.getActivation())
		return activations
 		
	def getLayers(self):
		return self.layers

	def getWeights(self):
		return self.weights

	def getOutput(self):
		return self.output

	def calculateCost(self,expectedOutput): 

		# compute the cost of a training example
		assert len(self.output) == len(expectedOutput)

		cost = 0
		for i in range(len(output)): 
			cost += (self.output[i] - expectedOutput[i])**2
		return cost


def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def normalizeMNIST(image):

	for i in range(len(image)):
		if(image[i] > 0):
			image[i] = (255 - float(image[i]))/255
	return image

def convertToOutput(number):

	print number
	

# ------ run below here ------- 

mndata = MNIST('./mnist')
images, labels = mndata.load_training()


test_image_index = 10

testImg = normalizeMNIST(images[test_image_index])
testImgValue = labels[test_image_index]
# print 'test image:',testImgValue

convert = convertToOutput(testImgValue)

# input = [] # empty input (to run network with random input values)
input = testImg

inputLayerSize = len(testImg)

# initialize the network
net = Network([inputLayerSize,16,16,10],sigmoid)
net.initializeNetwork()

# run the network
net.runNetwork(input)

# get the output
output = net.getOutput()
# print 'output: ', output

# compute the cost
expectedOutput = [0,0,0,0,0,1,0,0,0,0] # represents a 3
cost = net.calculateCost(expectedOutput)
# print 'cost:', cost








