Apologies for the slightly late submission - the large datasets took unbelievably long to run. Please note we submitted the code to the repo on time. We were simply waiting for jupyter to finish running on all these large datasets.. 

Our decision tree works by using functions (represented by functools) to test for either equality, greater than, orless than relationships. At each node, we generate every possible test for a node (without generating redundant tests if there are multiple values in a row with the same label). We obtain the informmation gain or gain ratio of each  and return the feature and function that yields the information gain/gain ratio. 

Our experimental component is contained in metrics.py. It takes stratified partitions of the training data (with overlap), calculates the gain ratio for each partition, and returns the average gain ratio across all partitions. In this way, we hoped to account for the importance of variability in training data selection in order to improve the generalizability of our algorithm.

