'https://medium.com/dev-genius/python-for-experienced-programmers-a2ee334ce62f'
#let's use dictionaries for our data
import math
import random
def find_indices(list, condition):
    return [i for i, elem in enumerate(list) if condition(elem)]


def crossval(labels, folds, iterations):

    pos_lab = find_indices(labels, lambda x: x>0)
    neg_lab = find_indices(labels, lambda x: x<=0)
    pos_per_fold = math.floor(len(pos_lab) / folds)
    neg_per_fold = math.floor(len(neg_lab) / folds)
    pos_remainder = math.modf(len(pos_lab) / folds)
    neg_remainder = math.modf(len(neg_lab) / folds)

    pos_lab_shuff = pos_lab[:]
    neg_lab_shuff = neg_lab[:]
    random.shuffle(pos_lab_shuff)
    random.shuffle(neg_lab_shuff)
    data = []
    for x in range(1,folds):
        pos_count =  pos_per_fold + min(1, pos_remainder)
        neg_count =  neg_per_fold + min(1, neg_remainder)
        data[x] = [pos_lab_shuff[1:pos_count] + neg_lab_shuff[1:neg_count]]
        pos_per_fold = max(0, pos_per_fold - 1)
        neg_per_fold = max(0, neg_per_fold - 1)
        del pos_lab_shuff[1:pos_count]
        del neg_lab_shuff[1:neg_count]
    return data





