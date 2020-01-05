# Cmpe 480 - Introduction to Artificial Intelligence
# Project 3 - Decision Tree Implementation
# Seyfi Kutay Kılıç

import sys
import collections
import random
import math

import statistics
import matplotlib.pyplot as plt

DATA_FILE = sys.argv[1]
ITERATION_COUNT = 10

def main():
	data = readData()
	# 10 times with information gain formula
	decision_trees_with_information_gain = []
	for i in range(ITERATION_COUNT):
		training_data, validation_data, test_data = shuffleAndSplitToTrainingValidationTest(data)
		decision_tree = DecisionTree(training_data, validation_data, test_data, EntropyCalculator.calculateEntropyWithInformationGainFormula)
		decision_trees_with_information_gain.append(decision_tree)
	# 10 times with gini impurity formula
	decision_trees_with_gini_impurity = []
	for i in range(ITERATION_COUNT):
		training_data, validation_data, test_data = shuffleAndSplitToTrainingValidationTest(data)
		decision_tree = DecisionTree(training_data, validation_data, test_data, EntropyCalculator.calculateEntropyWithGiniImpurityFormula)
		decision_trees_with_gini_impurity.append(decision_tree)
	# report the results
	OutputUtility.reportLossInTestSet(decision_trees_with_information_gain, decision_trees_with_gini_impurity)
	OutputUtility.reportMeanAndVariancesOfLosses(decision_trees_with_information_gain, decision_trees_with_gini_impurity)
	OutputUtility.plotLossRates(decision_trees_with_information_gain, decision_trees_with_gini_impurity)

# reads the data and convert it to Datum objects
def readData():
	data = []
	with open(DATA_FILE, "r") as f:
		for line in f.read().splitlines():
			if len(line) > 1: # if not an empty line
				row = line.split(',')
				data.append(Datum([float(num) for num in row[:-1]], row[-1]))
	return data

# Split the data into training(20%), validation(40%), and test (40%)
def shuffleAndSplitToTrainingValidationTest(data):
	random.shuffle(data)
	tenpercent_n = len(data) // 10
	training_data = data[ : 2*tenpercent_n]
	validation_data = data[2*tenpercent_n : 6*tenpercent_n]
	test_data = data[6*tenpercent_n : ]
	return (training_data, validation_data, test_data)


class DecisionTree:
	def __init__(self, training_data, validation_data, test_data, entropy_calculator_function):
		self.training_data = training_data
		self.validation_data = validation_data
		self.test_data = test_data
		self.entropy_calculator_function = entropy_calculator_function

		self.training_loss_of_depths = []
		self.validation_loss_of_depths = []

		self.root = DecisionTreeNode(training_data[:])
		self.buildTree()

	# Build the tree in bfs order
	# If the splitting is not rational with respect to validation data then it starts to stop the splittings
	def buildTree(self):
		current_error_rate = self.computeValidationError()
		q = collections.deque()
		q.append(self.root)
		while q:
			self.training_loss_of_depths.append(self.computeTrainingError())
			self.validation_loss_of_depths.append(self.computeValidationError())
			# do layer by layer traversel since I am also interested in the loss rates of layers
			level_size = len(q)
			for i in range(level_size):
				node = q.popleft()
				if node.isSuitableToSplit():
					param_index, param_value = self.findBestSplit(node.data)
					node.split_param_index, node.split_param_value = param_index, param_value
					left_data, right_data = self.splitWithAttributeAndValue(node.data, param_index, param_value)
					node.left, node.right = DecisionTreeNode(left_data), DecisionTreeNode(right_data)
					new_error_rate = self.computeValidationError()
					if new_error_rate < current_error_rate: # We can hold the split
						current_error_rate = new_error_rate
						q.append(node.left)
						q.append(node.right)
					else: # The split is not rational because error rate has increased
						node.left, node.right = None, None # Recover children

	# Finds the best split paramater and it's value which gives min entropy
	def findBestSplit(self, data):
		#print([vars(datum) for datum in data])
		best_entropy = 999
		best_param_index = None
		best_param_value = None
		for param_index in range(len(data[0].params)):
			for datum in data:
				left_data, right_data = self.splitWithAttributeAndValue(data, param_index, datum.params[param_index])
				entropy = self.computeEntropyAfterSplitting(left_data, right_data)
				if entropy < best_entropy:
					best_entropy = entropy
					best_param_index = param_index
					best_param_value = datum.params[param_index]
		return (best_param_index, best_param_value)

	# Split the data into two parts with respect to the given splitting point
	def splitWithAttributeAndValue(self, data, param_index, param_value):
		left_data = []
		right_data = []
		for datum in data:
			if datum.params[param_index] < param_value:
				left_data.append(datum)
			else:
				right_data.append(datum)
		return (left_data, right_data)

	# Compute the average entropy of left and right data
	def computeEntropyAfterSplitting(self, left_data, right_data):
		left_entropy = self.entropy_calculator_function(left_data)
		right_entropy = self.entropy_calculator_function(right_data)
		total_size = len(left_data) + len(right_data)
		return (len(left_data) * left_entropy + len(right_data) * right_entropy) / total_size

	def computeValidationError(self):
		return DecisionTreeEvaluator.computeErrorPercentage(self.root, self.validation_data)

	def computeTrainingError(self):
		return DecisionTreeEvaluator.computeErrorPercentage(self.root, self.training_data)

# An extended Binary Tree Node to be used in the decision tree
class DecisionTreeNode:
	def __init__(self, data):
		self.data = data
		self.pluratiy_class = self.findPluratiyClass()
		self.left = None
		self.right = None
		self.split_param_index = None
		self.split_param_value = None 
	
	# Returns the class that is most frequent in the data of node	
	def findPluratiyClass(self):
		classes = [datum.classs for datum in self.data]
		return max(set(classes), key = classes.count)

	# The node is suitable to split if it has at least two nodes and has different class datums
	def isSuitableToSplit(self):
		return len(self.data) >= 2 and len(set([datum.classs for datum in self.data])) > 1

# Object representation of datums
class Datum:
	def __init__(self, params, classs):
		self.params = params
		self.classs = classs

# Utility function to compute the errors, predict the result, etc by using a decision tree
class DecisionTreeEvaluator:
	@staticmethod
	def computeErrorPercentage(decision_tree_root, data):
		error_count = 0
		for datum in data:
			actual_result = datum.classs
			predicted_result = DecisionTreeEvaluator.predictResult(decision_tree_root, datum.params)
			if actual_result != predicted_result:
				error_count += 1
		return (error_count / len(data)) * 100

	@staticmethod
	def predictResult(decision_tree_root, params):
		current_node = decision_tree_root
		# while it is not a leaf
		while current_node.left is not None: # note that, if the left is null then right is also null and vice versa
			if params[current_node.split_param_index] < current_node.split_param_value:
				current_node = current_node.left
			else:
				current_node = current_node.right
		return current_node.pluratiy_class

# This module contains two different entropy calculation formulas, which are:
# Information Gain and Gini Impurity
class EntropyCalculator:
	@staticmethod
	def calculateEntropyWithInformationGainFormula(data):
		classes = [datum.classs for datum in data]
		class_set = set(classes)
		if len(class_set) < 2:
			return 0
		entropy = 0
		for classs in class_set:
			proportion = classes.count(classs) / len(data)
			entropy -= proportion * math.log2(proportion)
		return entropy

	@staticmethod
	def calculateEntropyWithGiniImpurityFormula(data):
		classes = [datum.classs for datum in data]
		class_set = set(classes)
		if len(class_set) < 2:
			return 0
		entropy = 1
		for classs in class_set:
			proportion = classes.count(classs) / len(data)
			entropy -= proportion * proportion
		return entropy

# This module contains utility functions to print and plot the necessary outputs, such as:
# Loss rates of iteration, mean and variance of test data losses, and plots of loss rates with respect to decision tree depth
class OutputUtility:
	@staticmethod
	def reportLossInTestSet(decision_trees_with_information_gain, decision_trees_with_gini_impurity):
		information_gain_losses, gini_impurity_losses = OutputUtility.getLossPercentagesInTestSet(decision_trees_with_information_gain, decision_trees_with_gini_impurity)
		for i in range(len(information_gain_losses)):
			loss_percantage = information_gain_losses[i]
			print(f"The loss percentage of information gain technique in iteration {i+1} is {loss_percantage}%")
		for i in range(len(gini_impurity_losses)):
			loss_percantage = gini_impurity_losses[i]
			print(f"The loss percentage of gini impurity technique in iteration {i+1} is {loss_percantage}%")

	@staticmethod
	def reportMeanAndVariancesOfLosses(decision_trees_with_information_gain, decision_trees_with_gini_impurity):
		information_gain_losses, gini_impurity_losses = OutputUtility.getLossPercentagesInTestSet(decision_trees_with_information_gain, decision_trees_with_gini_impurity)
		information_gain_mean = statistics.mean(information_gain_losses)
		information_gain_variance = statistics.variance(information_gain_losses)
		gini_impurity_mean = statistics.mean(gini_impurity_losses)
		gini_impurity_variance = statistics.variance(gini_impurity_losses)
		print(f"The mean of the loss with using information gain is {information_gain_mean}%")
		print(f"The mean of the loss with using gini impurity is {gini_impurity_mean}%")
		print(f"The variance of the loss with using information gain is {information_gain_variance}%")
		print(f"The variance of the loss with using gini impurity is {gini_impurity_variance}%")

	# Helper method to get the loss rates of test data
	@staticmethod 
	def getLossPercentagesInTestSet(decision_trees_with_information_gain, decision_trees_with_gini_impurity):
		information_gain_losses = []
		gini_impurity_losses = []
		for i in range(len(decision_trees_with_information_gain)):
			decision_tree = decision_trees_with_information_gain[i]
			loss_percantage = DecisionTreeEvaluator.computeErrorPercentage(decision_tree.root, decision_tree.test_data)
			information_gain_losses.append(loss_percantage)
		for i in range(len(decision_trees_with_gini_impurity)):
			decision_tree = decision_trees_with_gini_impurity[i]
			loss_percantage = DecisionTreeEvaluator.computeErrorPercentage(decision_tree.root, decision_tree.test_data)
			gini_impurity_losses.append(loss_percantage)
		return (information_gain_losses, gini_impurity_losses)

	@staticmethod
	def plotLossRates(decision_trees_with_information_gain, decision_trees_with_gini_impurity):
		plt.title('Error Percentages')

		plt.subplot(2, 2, 1)
		plt.xlabel('Training Loss With Respect to Depth (Information Gain)')
		for i in range(len(decision_trees_with_information_gain)):
			decision_tree = decision_trees_with_information_gain[i]
			plt.plot(decision_tree.training_loss_of_depths)

		plt.subplot(2, 2, 2)
		plt.xlabel('Training Loss With Respect to Depth (Gini Impurity)')
		for i in range(len(decision_trees_with_gini_impurity)):
			decision_tree = decision_trees_with_gini_impurity[i]
			plt.plot(decision_tree.training_loss_of_depths)

		plt.subplot(2, 2, 3)
		plt.xlabel('Validation Loss With Respect to Depth (Information Gain)')
		for i in range(len(decision_trees_with_information_gain)):
			decision_tree = decision_trees_with_information_gain[i]
			plt.plot(decision_tree.validation_loss_of_depths)

		plt.subplot(2, 2, 4)
		plt.xlabel('Validation Loss With Respect to Depth (Gini Impurity)')
		for i in range(len(decision_trees_with_gini_impurity)):
			decision_tree = decision_trees_with_gini_impurity[i]
			plt.plot(decision_tree.validation_loss_of_depths)

		plt.subplots_adjust(hspace=0.5)
		plt.show()


main()