# Cmpe 480 - Introduction to Artificial Intelligence
# Project 3
# Seyfi Kutay Kılıç

import collections
import random
import math

DATA_FILE = "iris.data"

def main():
	data = readData()
	# 10 times with information gain formula
	for i in range(10):
		training_data, validation_data, test_data = shuffleAndSplitToTrainingValidationTest(data)
		decision_tree = DecisionTree(training_data, validation_data, EntropyCalculator.calculateEntropyWithInformationGainFormula)
		print(DecisionTreeEvaluator.computeErrorRate(decision_tree, test_data))
	print("***")
	# 10 times with information gini impurity formula
	for i in range(10):
		training_data, validation_data, test_data = shuffleAndSplitToTrainingValidationTest(data)
		decision_tree = DecisionTree(training_data, validation_data, EntropyCalculator.calculateEntropyWithGiniImpurityFormula)
		print(DecisionTreeEvaluator.computeErrorRate(decision_tree, test_data))

def readData():
	data = []
	with open(DATA_FILE, "r") as f:
		for line in f.read().splitlines():
			row = line.split(',')
			data.append(Datum([float(num) for num in row[:-1]], row[-1]))
	return data

def shuffleAndSplitToTrainingValidationTest(data):
	random.shuffle(data)
	tenpercent_n = len(data) // 10
	training_data = data[ : 2*tenpercent_n]
	validation_data = data[2*tenpercent_n : 6*tenpercent_n]
	test_data = data[6*tenpercent_n : ]
	return (training_data, validation_data, test_data)

class DecisionTree:
	def __init__(self, training_data, validation_data, entropy_calculator_function):
		self.training_data = training_data
		self.validation_data = validation_data
		self.entropy_calculator_function = entropy_calculator_function
		self.root = DecisionTreeNode(training_data[:])
		self.buildTree()

	def buildTree(self):
		current_error_rate = self.computeValidationError()
		q = collections.deque()
		q.append(self.root)
		while q:
			level_size = len(q)
			for i in range(level_size):
				node = q.popleft()
				if node.isSuitableToSplit():
					param_index, param_value = self.findBestSplit(node.data)
					node.split_param_index, node.split_param_value = param_index, param_value
					left_data, right_data = self.computeSplitWithAttributeAndValue(node.data, param_index, param_value)
					node.left, node.right = DecisionTreeNode(left_data), DecisionTreeNode(right_data)
					new_error_rate = self.computeValidationError()
					if new_error_rate < current_error_rate: # We can hold the split
						current_error_rate = new_error_rate
						q.append(node.left)
						q.append(node.right)
					else: # The split is not rational because error rate has increased
						node.left, node.right = None, None # Recover children


	def findBestSplit(self, data):
		#print([vars(datum) for datum in data])
		best_entropy = 999
		best_param_index = None
		best_param_value = None
		for param_index in range(len(data[0].params)):
			for datum in data:
				left_data, right_data = self.computeSplitWithAttributeAndValue(data, param_index, datum.params[param_index])
				entropy = self.computeEntropyAfterSplitting(left_data, right_data)
				if entropy < best_entropy:
					best_entropy = entropy
					best_param_index = param_index
					best_param_value = datum.params[param_index]
		return (best_param_index, best_param_value)

	def computeSplitWithAttributeAndValue(self, data, param_index, param_value):
		left_data = []
		right_data = []
		for datum in data:
			if datum.params[param_index] < param_value:
				left_data.append(datum)
			else:
				right_data.append(datum)
		return (left_data, right_data)

	def computeEntropyAfterSplitting(self, left_data, right_data):
		left_entropy = self.entropy_calculator_function(left_data)
		right_entropy = self.entropy_calculator_function(right_data)
		total_size = len(left_data) + len(right_data)
		return (len(left_data) * left_entropy + len(right_data) * right_entropy) / total_size

	def computeValidationError(self):
		return DecisionTreeEvaluator.computeErrorRate(self, self.validation_data)


class DecisionTreeNode:
	def __init__(self, data):
		self.data = data
		self.pluratiy_class = self.findPluratiyClass()
		self.left = None
		self.right = None
		self.split_param_index = None
		self.split_param_value = None 
		
	def findPluratiyClass(self):
		classes = [datum.classs for datum in self.data]
		return max(set(classes), key = classes.count)

	def isSuitableToSplit(self):
		return len(self.data) > 2 and len(set([datum.classs for datum in self.data])) > 1


class Datum:
	def __init__(self, params, classs):
		self.params = params
		self.classs = classs

class DecisionTreeEvaluator:
	@staticmethod
	def computeErrorRate(decision_tree, data):
		error_count = 0
		for datum in data:
			actual_result = datum.classs
			predicted_result = DecisionTreeEvaluator.predictResult(decision_tree, datum.params)
			if actual_result != predicted_result:
				error_count += 1
		return error_count / len(data)

	@staticmethod
	def predictResult(decision_tree, params):
		current_node = decision_tree.root
		# while it is not a leaf
		while current_node.left is not None: # note that, if the left is null then right is also null and vice versa
			if params[current_node.split_param_index] < current_node.split_param_value:
				current_node = current_node.left
			else:
				current_node = current_node.right
		return current_node.pluratiy_class

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


main()