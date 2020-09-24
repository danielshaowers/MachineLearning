from typing import Tuple

from numpy import array

from P1.mldata import Example, ExampleSet


class Observation(Example):

	@property
	def label(self):
		return self[-1]

	@property
	def features(self):
		return self[:-1]


class ObservationSet(ExampleSet):

	@property
	def observations(self) -> Tuple[Observation]:
		return tuple(Observation(e) for e in self.examples)

	def to_numpy_array(self, mapper=None):
		return array(self.to_float(mapper))

	def __getitem__(self, item):
		return Observation(item)
