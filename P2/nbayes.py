from P2 import mldata

from algorithm import Model


class NaiveBayes(Model):

	def __init__(self):
		# Add any additional model parameters
		super(NaiveBayes, self).__init__()

	def train(self, data: mldata.ExampleSet):
		pass

	def predict(self, data: mldata.ExampleSet):
		pass
