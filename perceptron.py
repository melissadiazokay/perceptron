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


class Run(Network):

	def __init__(self,net,dataset,verbose = False):
			
		self.net = net
		self.dataset = dataset
		self.verbose = verbose


	def calculateCost(self,output,expectedOutput): 

		# compute the cost of a training example
		assert len(output) == len(expectedOutput)

		cost = 0
		for i in range(len(output)): 
			cost += (output[i] - expectedOutput[i])**2
		return cost

	def computeTotalCost(self):
		
		images = self.dataset['images']
		labels = self.dataset['labels']

		totalCost = 0
		for i,image in enumerate(images):
			result = self.run(image,labels[i])
			totalCost += result['cost']
		totalCost = totalCost / len(images)

		return totalCost


	def run(self,image,label):

		# run a single image example
		
		image = self.normalizeMNIST(image)

		if(self.verbose): print 'image label:',label

		# run the network
		self.net.runNetwork(image)

		# get the output
		output = self.net.getOutput()
		if(self.verbose): print 'output:', self.decodeOutput(output)

		# compute the cost
		expectedOutput = self.encodeOutput(label)
		# print 'expected output:', expectedOutput
		cost = self.calculateCost(output,expectedOutput)
		if(self.verbose): print 'cost:', cost

		return { 'output' : output, 'cost' : cost }


	def encodeOutput(self,number):

		output = []

		for i in range(10): 
			if i == number:
				output.append(1)
			else: output.append(0)

		return output
		
	def decodeOutput(self,outputLayerActivations):

		greatest_num = 0
		for i in range(len(outputLayerActivations)): 
			if outputLayerActivations[i] > greatest_num: 
				greatest_num = outputLayerActivations[i]
				guess = i
		return guess

	def normalizeMNIST(self,image):

		for i in range(len(image)):
			if(image[i] > 0):
				image[i] = (255 - float(image[i]))/255
		return image
		

def sigmoid(x):
	return 1 / (1 + math.exp(-x))





# ------ run below here -------

# import dataset
mndata = MNIST('./mnist')
images, labels = mndata.load_training()

input_layer_size = len(images[0])
data_chunk_size = 10

# initialize the network
net = Network([input_layer_size,16,16,10],sigmoid)
net.initializeNetwork()

# run the network
run = Run(net, {
	'images' : images[:data_chunk_size],
	'labels' : labels[:data_chunk_size]
},True)

cost = run.computeTotalCost()
print 'Total cost:', cost

# run.run(images[0],labels[0])
# run.run(images[1],labels[1])
# run.run(images[2],labels[2])
# run.run(images[3],labels[3])
# run.run(images[4],labels[4])







