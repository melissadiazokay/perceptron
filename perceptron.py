import random
import math
from mnist import MNIST

class Neuron(): 

	def __init__(self,activation,bias):

		self.activation = activation
		self.bias = bias

	def think(self,weights,activations,activationFunction, verbose = False):

		# update own activation as per activationFunction( sum( weights_previous_layer * activations_previous layer) + bias )
		# ---
		# weights (list) - all of the weights from previous layer
		# activations (list) - all of the activations from previous layer
		# activationFunction (function) - activation function from Network
		# * length of weights and activations must be the same

		sigma = 0
		for i in range(len(activations)):
			sigma += weights[i] * activations[i]

		newActivation = activationFunction(sigma + self.bias)

		if(verbose): 
			print('sigma:',sigma,'\nbias:',self.bias,'\nactivation:',newActivation)

		self.activation = newActivation

	def getActivation(self): return self.activation

	def getBias(self): return self.bias

	def setActivation(self,activation): 
		self.activation = activation


class Network(Neuron): 

	def __init__(self,structure,activationFunction): 

		self.structure = structure
		self.activationFunction = activationFunction
		self.layers = []
		self.weights = []
		self.biases = []
		self.input = []
		self.output = []

	def initializeNetwork(self):

		# create a new network with random weights and biases
		# activations are initialized to random float between 0 and 1
		# weights are initialized to random float between -1 and 1
		# biases are initialized to random float between -1 and 1

		# all the input weights to all neurons indexed by [layer index][neuron index]
		self.weights = [] * len(self.structure)
		self.biases = [] * len(self.structure)

		# initialize network structure, weights, and biases 
		for i, n_neurons in enumerate(self.structure):
			layer = []
			self.weights.append([])
			self.biases.append([])
			for j in range(n_neurons):
				self.initializeWeights(i)
				bias = random.uniform(-1,1)
				self.biases[i].append(bias)
				layer.append(Neuron(random.random(),bias))

			self.layers.append(layer)

	def initializeWeights(self,layerIndex):

		# initialize the weights object to random values
		# 
		# layerIndex (int) - index of layer who's neurons' weights will be initialized

		if(layerIndex == 0): # first layer does not have input weights - fill with placeholders
			self.weights[0].append([])
		else: # fill with input weights corresponding to neurons from the **last** layer
			arr = []
			for k in range(self.structure[layerIndex - 1]):
				arr.append(random.uniform(-1,1))
			self.weights[layerIndex].append(arr)

	def runNetwork(self, input = [], verbose = False):

		# set the input layer activations and feed forward
		# - if no input activations are provided, random values will be used
		#
		# input (list) (optional) - a list of input layer activations

		verbose = False # disable logging
		
		if(len(input) > 0): 
			assert len(input) == len(self.layers[0])
			self.input = input
			for i in enumerate(self.layers[0]):
				self.layers[0][i[0]].setActivation(self.input[i[0]])
		self.feedForward(verbose)

	def feedForward(self, verbose = False):

		# feed input forward through all layers of the network

		for i,layer in enumerate(self.layers):
			if(i == 0): continue # skip the input layer
			activations = self.getLayerActivations(i-1)
			for j,neuron in enumerate(layer):
				weights = self.weights[i][j]
				if(verbose): print('\nlayer:',i,'neuron:',j)
				neuron.think(weights,activations,self.activationFunction,verbose)
		self.output = self.getLayerActivations(len(self.layers) - 1)

	def updateBias(self,layerIndex,neuronIndex,biasValue):

		# update a particular neuron's bias value in the Network's AND Neuron's bias properties
		# 
		# layerIndex (int) - index of layer
		# neuronIndex (int) - index of Neuron in layer
		# biaValue (float) - the new bias value

		self.biases[layerIndex][neuronIndex] = biasValue
		self.layers[layerIndex][neuronIndex].bias = biasValue
			
	def getLayerActivations(self,layerIndex):

		activations = []
		for neuron in self.layers[layerIndex]:
			activations.append(neuron.getActivation())
		return activations
 		
	def getLayers(self): return self.layers

	def getWeights(self): return self.weights

	def getBiases(self): return self.biases

	def getOutput(self): return self.output


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

	def run(self,input,label = False):

		# pass a single image through the network
		#
		# input (list) - input data
		# label (varies) (optional) - input data label
		
		input = self.normalizeMNIST(input)

		# run the network
		self.net.runNetwork(input, self.verbose)

		# get the output
		output = self.net.getOutput()

		# compute the cost
		expectedOutput = self.encodeOutput(label)
		# print( 'expected output:', expectedOutput )
		cost = self.calculateCost(output,expectedOutput)

		if(self.verbose): 
			print('\nimage label:', label, '\noutput:', self.decodeOutput(output), '\ncost:', cost )

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


# ------ run below here -------

# import dataset
mndata = MNIST('./mnist')
images, labels = mndata.load_training()

# prepare input data
input_layer_size = len(images[0])
data_chunk_size = 10
def sigmoid(x): return 1 / (1 + math.exp(-x))

# initialize the network
net = Network([input_layer_size,16,16,10],sigmoid)
net.initializeNetwork()

# run the network
run = Run(net, {
	'images' : images[:data_chunk_size],
	'labels' : labels[:data_chunk_size]
},False)

# compute the total cost
cost = run.computeTotalCost()
print( '\nTotal cost:', cost )


# run.run(images[0],labels[0])
# run.run(images[1],labels[1])
# run.run(images[2],labels[2])
# run.run(images[3],labels[3])
# run.run(images[4],labels[4])





