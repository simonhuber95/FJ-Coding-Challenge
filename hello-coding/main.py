from datagen import genererate_data
import keras
from keras import layers, callbacks
import random
import numpy as np


def shuffle_lists(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b

def get_data(n, d):
    # Get Training data
    true_data, false_data  = genererate_data(n, d)
    y_true = [1] * len(true_data)
    y_false = [0] * len(false_data)
    x = true_data + false_data
    y = y_true + y_false
    x, y = shuffle_lists(x,y)
    return x, y

# ToDO: Write a Classifier in Keras and train it on the data set.
# Try for an accuracy of >95% in the test set

# Create model
model = keras.Sequential()
model.add(keras.Input(shape=(2,)))
model.add(layers.Dense(5, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(1, activation = 'sigmoid'))


# Compile modes with RMSProp and BinaryCrossEntropy
model.compile(optimizer="rmsprop",loss='BinaryCrossentropy', metrics=['accuracy'])

# Train model until early stopping is reached
train_x, train_y = get_data(1000, 0.5)

# Add EarlyStopping callback on validator loss. Stops the training when model would overfit
es = callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience = 1)
# Train the model with a validation split of 10% and stop on the callback or after 100 epochs
model.fit(x=np.array(train_x), y= np.array(train_y), validation_split=0.1, epochs=100, shuffle= True, callbacks = [es])

# Evaluate model
eval_x, eval_y = get_data(100, 0.5)
model.evaluate(x = eval_x, y = eval_y)
