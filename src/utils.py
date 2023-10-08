import os

# 
def second2date(seconds):
    """ 
    input
    --------------------
    seconds (int)

    output
    --------------------
    date (str)
    """

    h = seconds // 3600
    m = (seconds - 3600*h) // 60
    s = seconds - 3600*h - 60*m

    return f'{h:0>2}:{m:0>2}:{s:0>2}'


# 
def make_folders(path):
    """ 
    input
    --------------------
    path (str)
    """
    os.mkdir(path)
    os.mkdir(path+'/cv')
    os.mkdir(path+'/train')
    os.mkdir(path+'/test')