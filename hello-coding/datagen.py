import random

def genererate_data(n, distr = None):
    """
    Generates Data Points from the distribution. 
    Args:
        n (int): number of points which are drawn in total
        distr (float): balance between correct and incorrect data points (y_true and y_false)
    
    Returns:
        true_data, false_data (list): Returns two Lists filled with tuples of x and y coordinates between [0,1]
    """
    rand_percent = distr if distr and 0<= distr <= 1 else random.random()
    true_data = [(round(random.randrange(50, 150)/100 -1, 2), round(random.randrange(50, 150)/100 - 1, 2)) for x in range(round(n*rand_percent))]
    false_data = [(round(random.choice([random.randrange(0, 50)/100 - 1, random.randrange(150, 200)/100 - 1]),2), round(random.choice([random.randrange(0, 50)/100 - 1, random.randrange(150, 200)/100 - 1]),2)) for x in range(round(n*(1-rand_percent)))]
    return true_data, false_data



