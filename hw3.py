# Cmpe 480 - Introduction to Artificial Intelligence
# Project 3
# Seyfi Kutay Kılıç

import random
import math

DATA_FILE = "iris.data"
ENTROPY_LOSS_TRESHOLD = 0.1

CLASS_SET = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}

def main():
	data = readData()
	training_data, validation_data, test_data = shuffleAndSplitToTrainingValidationTest(data)
	DecisionTree(training_data, validation_data, computeEntropyWithInformationGainFormula)

def readData():
	data = []
	with open(DATA_FILE, "r") as f:
		for line in f.read().splitlines():
			row = line.split(',')
			data.append({ "params" : [float(num) for num in row[:-1]], "class" : row[-1]})
	return data

def shuffleAndSplitToTrainingValidationTest(data):
	random.shuffle(data)
	tenpercent_n = len(data) // 10
	training_data = data[ : 2*tenpercent_n]
	validation_data = data[2*tenpercent_n : 6*tenpercent_n]
	test_data = data[6*tenpercent_n : ]
	return (training_data, validation_data, test_data)

def computeEntropyWithInformationGainFormula(data):
	entropy = 0
	classes = [row["class"] for row in data]
	for classs in CLASS_SET:
		count = classes.count(classs)
		if count != 0 and count != len(data):
			proportion = count / len(classes)
			entropy -= proportion * math.log2(proportion)
			entropy -= (1 - proportion) * math.log2(1 - proportion)
	return entropy

class DecisionTree:
	def __init__(self, training_data, validation_data, entropy_calculator_function):
		self.entropy_calculator_function = entropy_calculator_function
		self.root = self.generateTree(training_data[:], validation_data[:])

	def generateTree(self, training_data, validation_data):
		root = DecisionTreeNode
		()
		training_entropy = self.entropy_calculator_function(training_data)
		validation_entropy = self.entropy_calculator_function(validation_data)
		split_param, split_point = self.findBestSplit(training_data)
		training_left_data, training_right_data, training_entropy_after_split = self.splitWithParamAndPoint(training_data, split_param, split_point)
		validation_left_data, validation_right_data, validation_entropy_after_split = self.splitWithParamAndPoint(validation_data, split_param, split_point)
		entropy_loss_of_validation_data = validation_entropy - validation_entropy_after_split
		if entropy_loss_of_validation_data < ENTROPY_LOSS_TRESHOLD: # This node must be a leaf node
			root.result_class = self.computeResultOfLeafNode(training_data)
		else: # Continue to branching
			root.split_param = split_param
			root.split_point = split_point
			root.left = generateTree(training_left_data, validation_left_data)
			root.right = generateTree(training_right_data, validation_right_data)
		return root

	def findBestSplit(self, training_data):
		best_entropy = 999
		best_param = None
		best_point = None
		for param_index in range(len(training_data[0]["params"])):
			for row in training_data:
				left_data, right_data, entropy = self.splitWithParamAndPoint(training_data, param_index, row["params"][param_index])
				if entropy < best_entropy:
					best_entropy = entropy
					best_param = param_index
					best_point = row["params"][param_index]
		return (best_param, best_point)

	def splitWithParamAndPoint(self, data, param_index, param_point):
		left_data = []
		right_data = []
		for row in data:
			if row["params"][param_index] < param_point:
				left_data.append(row)
			else:
				right_data.append(row)
		entropy_after_split = self.computeEntropyAfterSplitting(left_data, right_data)
		return (left_data, right_data, entropy_after_split)

	def computeEntropyAfterSplitting(self, left_data, right_data):
		left_entropy = self.entropy_calculator_function(left_data)
		right_entropy = self.entropy_calculator_function(right_data)
		total_size = len(left_data) + len(right_data)
		return (len(left_data) * left_entropy + len(right_data) * right_entropy) + total_size

	def computeResultOfLeafNode(self, data):
		classes = [row["class"] for row in data]
		return max(CLASS_SET, key = classes.count)

class DecisionTreeNode:
	def __init__(self):
		self.split_param = None
		self.split_point = None 
		self.left = None
		self.right = None
		self.result_class = None


main()