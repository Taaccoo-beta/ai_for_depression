import numpy as np 


def get_accuracy(result_list,test_dataset):
    ans_to_ix = test_dataset.ans_to_ix 
    result_list = np.array(result_list)

    return (ans_to_ix == result_list).float().sum() / ans_to_ix.__len__()

