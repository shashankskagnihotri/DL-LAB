import numpy as np

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4

def one_hot_new(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    #print("n_classes: ", n_classes)
    #one_hot_labels = np.zeros(labels.shape + (n_classes,))

    one_hot_labels = np.array([], dtype = 'float64')
    one_hot_labels = one_hot_labels.reshape((0,5))

    for label in labels:
        if label == LEFT:
            one_hot_labels = np.append(one_hot_labels, [[0.0, 1.0, 0.0, 0.0, 0.0]], axis = 0)
        elif label == RIGHT:
            one_hot_labels = np.append(one_hot_labels, [[0.0, 0.0, 1.0, 0.0, 0.0]], axis = 0)
        elif label == ACCELERATE:
            one_hot_labels = np.append(one_hot_labels, [[0.0, 0.0, 0.0, 1.0, 0.0]], axis = 0)
        elif label == BRAKE:
            one_hot_labels = np.append(one_hot_labels, [[0.0, 1.0, 0.0, 0.0, 1.0]], axis = 0)
        else:
            one_hot_labels = np.append(one_hot_labels, [[1.0, 0.0, 0.0, 0.0, 0.0]], axis = 0)

        #print("one_hot_labels:", one_hot_labels)
    
    return one_hot_labels

def one_hot(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = 5
    #print("n_classes: ", n_classes)
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
        #print("labels: ", labels)
        #print("one_hot_labels[",c,"]: ", one_hot_labels[c])
    return one_hot_labels
    

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 


def action_to_id(a):
    """ 
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]): return LEFT               # LEFT : 1
    elif all(a == [-1.0, 1.0, 0.0]): return LEFT             # LEFT : 1
    elif all(a == [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif all(a == [1.0, 1.0, 0.0]): return RIGHT             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 1.0]): return BRAKE             # BRAKE: 4
    elif all(a == [-1.0, 0.0, 1.0]): return BREAK            # BREAK: 4
    elif all(a == [1.0, 0.0, 1.0]): return BREAK             # BREAK: 4
    elif all(a == [0.0, 0.0, 0.0]): return STRAIGHT          # STRAIGHT: 0
    else:       
        return ACCELERATE                                    # ACCELERATE : 3

def id_to_action(id):
    if id == LEFT:
        return np.array([-1.0, 0.0, 0.0])
    elif id == RIGHT:
        return np.array([1.0, 0.0, 0.0])
    elif id == ACCELERATE:
        return np.array([0.0, 1.0, 0.0])
    elif id == BRAKE:
        return np.array([0.0, 0.0, 1.0])
    else:
        return np.array([0.0, 0.0, 0.0])

def reshaped_history(x, history_length):

    #print("Shape of x", x.shape)
    reshaped = np.empty((x.shape[0], x.shape[1], x.shape[2], history_length))
    #print("Shape of Reshaped", reshaped.shape)
    #print("x:",x)

    for index in range(x.shape[0] - history_length):
        reshaped[index, :, :, :] = np.transpose(x[index: index + history_length, :, :, 0], (1, 2, 0))

    return reshaped
