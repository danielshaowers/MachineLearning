import mldata

import model


class NaiveBayes(model.Model):

	def __init__(self):
		# Add any additional model parameters
		super(NaiveBayes, self).__init__()

	def train(self, data: mldata.ExampleSet):
		pass

	def predict(self, data: mldata.ExampleSet):
		pass
