import numpy as np
import preprocess
def main():
    # load dataset
    # convert to numpy array
    # preprocess by normalizing
    dataset = load_dataset("random")
    labels = dataset[0]
    dataset = dataset[1:]
    nU = len(dataset[0]) - len(labels) # num unlabled
    # get labels for dataset
    dataset = preprocess.normalize(dataset)
    # done with preprocessing. now we do the fun stuff

    # COMPUTE WEIGHTS
    # first get sigmas
    # todo: check data if features are columns or rows
    guesses = makeGuess(np.random(len(dataset)), labels)
    sigmas = gfield_grad_descent(np.transpose(dataset), labels, guesses, step = 0.05)
    weights = weight_edges(labels, dataset, sigmas)
    fu = makeGuess(weights, labels)

# returns vector of f for unlabeled points
def makeGuess(weights, labels):
    diags = np.diag(np.sum(weights,
                           axis=1))  # might be wrong: sum for each entry in the list. since each list is one example. put into diagonal matrix
    delta = diags - weights  # combinatorial laplacian
    P = np.invert(delta) * weights  # P = D^-1 W, and f=Pf or something. gonna ignore this for now
    wLL, wLU, wUL, wUU = quadrate(weights, len(labels))
    return np.invert(diags - weights) * wUL * labels, P  # equation 5. f(i)

def weight_edges(labels, data, sigmas):
    # sum over every dimension using the sigmas
    wmatrix = np.zeros(len(data), len(data)) # equation 1: compute distance based on sigmas
    for ind,i in enumerate(data):
        for ind2, e in enumerate(data):
            temp = np.square(i-e) / np.square(sigmas) # should operate on two vectors of features
            wmatrix[ind][ind2] = np.exp(-np.sum(temp))
    return wmatrix

def harmonic_f(weights, diagonals):
    f = np.zeros(len(weights))
    for i,e in enumerate(weights): # runs through each example
        f[i] = np.sum(weights[i] * f(i)) # equation 3: but we don't know f(i)? gonna ignore this part for now too

# divide matrix into quadrants
def quadrate(matrix, split):
    wLL = matrix[0:split - 1][0:split - 1]
    wLU = matrix[0:split - 1][split: len(matrix)]
    wUL = matrix[split: len(matrix)][0:split - 1]
    wUU = matrix[split: len(matrix)][split: len(matrix)]
    return wLL, wLU, wUL, wUU

def gfield_grad_descent(dataset, labels, weights, sigmas, step = 0.05):
    # smooth P for minimization
    scores, p = makeGuess(weights, labels) # worried this line may be slow

    epsilon = 1/10
    Pbar = epsilon * (1 / len(dataset[0])) + (1 - epsilon) * p # len(dataset[0]) = 1/(l+u)
    pLL, pLU, pUL, pUU = quadrate(Pbar, len(labels))
    nU = len(dataset[0]) - len(labels)
    ul_data = dataset[:][len(labels):]
    # run separately for each feature dimension
    I = np.diag(np.ones(nU))

    for i,e in enumerate(dataset): # for each feature
        dP_dsig = np.zeros(len(dataset[0]), len(dataset[0]))
        dW_dsig = dP_dsig
        for ii, ee in enumerate(e): # for each exmaple
            for iii, eee in enumerate(e): # also for each example
                dW_dsig[ii][iii] = (2 * weights[ii][iii] * np.square(ee - eee)) / np.power(sigmas[i], 3)
        for ii, ee in enumerate(e):  # for each exmaple
            for iii, eee in enumerate(e):
                dP_dsig[ii][iii] = dW_dsig[ii][iii] - Pbar[ii][iii] * np.sum(dW_dsig[ii][len(labels):]) / np.sum(weights[ii][len(labels):]) # equation 14

        _, _, dPul_dsig, dPuu_dsig = quadrate(dP_dsig, len(labels))
        df_dsig = np.invert(I - pUU) * (dPuu_dsig * scores + dPul_dsig * labels)
        dH_dsig = (1/nU) * np.sum(df_dsig * (1 - scores) / scores) # this is the gradient
        sigmas[i] = sigmas[i] + step * dH_dsig



def load_dataset(name):
    if name == "name":
        data = np.array([np.round(np.random(10))], [np.random(20)], [np.random(20)]) # load dataset with that name
    if name == "name":
        data = 2
    return data