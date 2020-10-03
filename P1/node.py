from collections import Counter


class Node:
	"""A node with possibly a left and right child, internal data, and parent.
	"""

	def __init__(self, data=None, left=None, right=None, parent=None):
		self.left = left
		self.right = right
		self.parent = parent
		self.data = data

	def is_leaf(self):
		return not self.has_left_child() and not self.has_right_child()

	def has_left_child(self):
		return self.left is not None

	def has_right_child(self):
		return self.right is not None

	def get_max_depth(self, depth: int = 0) -> int:
		if self.has_left_child():
			depth = max(self.left.get_max_depth(depth + 1), depth)
		if self.has_right_child():
			depth = max(self.right.get_max_depth(depth + 1), depth)
		return depth

	def main_leaf_vals(self):
		myvals = self.get_leaf_vals([0])
		return Counter(myvals).most_common()

	def get_leaf_vals(self, leaves) -> list:
		if self.has_left_child():
			leaves = leaves + self.left.get_leaf_vals(leaves)
		if self.has_right_child():
			leaves = leaves + self.right.get_leaf_vals(leaves)
		if self.is_leaf():
			return [self.data]

	def get_tree_size(self, tree_size: int = 1) -> int:
		if self.has_left_child():
			tree_size += self.left.get_tree_size(tree_size + 1)
		if self.has_right_child():
			tree_size += self.right.get_tree_size(tree_size + 1)
		return tree_size
