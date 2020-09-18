from numpy import array

from mldata import Example, ExampleSet


class Observation(Example):

	@property
	def label(self):
		return self[-1]

	@property
	def features(self):
		return self[:-1]


class ObservationSet(ExampleSet):

	@property
	def observations(self):
		return self.examples

	def to_numpy_array(self, mapper=None):
		return array(self.to_float(mapper))

	def __getitem__(self, item):
		return Observation(item)
