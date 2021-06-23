import random

def genererate_data(n, distr = None):
    """
    Generates Data Points from the distribution
    Args:
        n (int): number of points which are drawn in total
        distr (float): balance between correct and incorrect data points (y_true and y_false)
    
    Returns:
        true_data, false_data (list): Returns two Lists filled with tuples of x and y coordinates between [0,1]
    """
    rand_percent = distr if distr and 0<= distr <= 1 else random.random()
    true_data = [(random.randrange(0, 50)/100,random.randrange(0, 50)/100) for x in range(round(n*rand_percent))]
    false_data = [(random.randrange(50, 100)/100,random.randrange(50, 100)/100) for x in range(round(n*(1-rand_percent)))]
    return true_data, false_data

