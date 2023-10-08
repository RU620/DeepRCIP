import numpy as np

# 
def seq2onehot(seq):
    """ 
    input
    --------------------
    seq (str) : RNA sequence with length N.

    output
    --------------------
    oh (list) : One-hot encoding with size 4x200.
    """
    oh = []
    for s in seq:
        if   s=='A': oh.append([1,0,0,0])
        elif s=='U': oh.append([0,1,0,0])
        elif s=='G': oh.append([0,0,1,0])
        elif s=='C': oh.append([0,0,0,1])
    while len(oh)<200: oh.append([0,0,0,0])
    return np.array(oh, dtype=np.float32).T.tolist()