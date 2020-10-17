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

A lengthy description of the experimental model is provided in `experiment.py`.

Google Colab was used to train all of the models as it provided accelerated
GPU capabilities across multiple notebooks simultaneously. Needless to say
, this sped up the training process considerably. The results for each part
of the assignment (parts a-c) are stored as JSON text files for reference.
