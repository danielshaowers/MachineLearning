import mldata

import model


class LogisticRegression(model.Model):

	def __init__(self):
		# Add any additional model parameters
		super(LogisticRegression, self).__init__()

	def train(self, data: mldata.ExampleSet):
		pass

	def predict(self, data: mldata.ExampleSet):
		pass
