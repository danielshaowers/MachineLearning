# Executing the Code
To run either module, run `python <module.py> --help`. The command line
arguments follow what is expected for the assignment, but are entered with
their corresponding argument name for readability.

# Notes on the Assignment
For naive Bayes, docstrings are provided for the class and the majority of
the methods. That along with the method and variable naming will hopefully
make the code fairly straightforward. The `metrics.probability()` function
plays a significant role in actually generating the model parameters. For
discretizing continuous variables, `functools.partial()` is used along with
`numpy.digitize()` to "train" one function per continuous feature these
functions are then stored and used during training and prediction for
discretization.

For log reg, we minimized conditional log likelihood with overfitting control by taking the derivative with respect to each feature, which provided the gradient w + sum_i(h_i(x)-y_i)xij. The performance was greatly improved by computing a vector for the summation term separately from the xij term. This way, we only needed to perform the summation once for each feature.  
A lengthy description of the experimental model is provided in `experiment.py`.

Google Colab was used to train all of the models as it provided accelerated
GPU capabilities across multiple notebooks simultaneously. Needless to say
, this sped up the training process considerably. The results for each part
of the assignment (parts a-c) are stored as JSON text files for reference.
