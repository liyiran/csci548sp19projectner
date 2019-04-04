import abc

class DITKModel(abc.ABC):

	# Any shared data strcutures or methods should be defined as part of the parent class.
	
	# A list of shared arguments should be defined for each of the following methods and replace (or precede) *args.
	
	# The output of each of the following methods should be defined clearly and shared between all methods implemented by members of the group. 
	############### NOTES ##############
	# AGREEMENT - For the abstract class, no parameters will be required. Every subclass will implement individual parameters for their methods
	#
	#
	@classmethod
	@abc.abstractmethod
	# return nothing
	def read_dataset(*args, **kwargs):
		pass

	@classmethod
	@abc.abstractmethod
	# return nothing
	def train(*args, **kwargs):
		pass

	@classmethod
	@abc.abstractmethod
	# return ??
	def predict(*args, **kwargs):
		pass

	@classmethod
	@abc.abstractmethod
	# print (precision, recall, f1)
	def evaluate(*args, **kwargs):
		pass


# Example pipeine and implementation
# def REALCLASS(DITKModel):
# 	def train(*args, **kwargs):
# 			pass
# a = REALCLASS1
# b = REALCLASS2

# for i in stuff:
# 	data = t.read_dataset()
# 	i.train(data)