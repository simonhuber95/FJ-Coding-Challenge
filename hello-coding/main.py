from datagen import genererate_data
import keras
from keras import layers, callbacks
import random
import numpy as np
import matplotlib.pyplot as plt


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
history = model.compile(optimizer="rmsprop",loss='BinaryCrossentropy', metrics=['accuracy'])

# Train model until early stopping is reached
train_x, train_y = get_data(10000, 0.5)

# Add EarlyStopping callback on validator loss. Stops the training when model would overfit
es = callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience = 1)
# Train the model with a validation split of 10% and stop on the callback or after 100 epochs
model.fit(x=np.array(train_x), y= np.array(train_y), validation_split=0.1, epochs=100, shuffle= True, callbacks = [es])

# Predict some values from distribution
pred_x, _ = get_data(10000, 0.5)
pred_y = model.predict(pred_x)

# Predict from random values the model has never seen
# pred_x = [(round(random.randrange(0, 200)/100 -1, 2), round(random.randrange(0, 200)/100 -1, 2)) for x in range(10000)]
# pred_y = model.predict(pred_x)

# Filter the predictions depending on the output of the MLP
delta = 0.05
true_preds, _ = zip(*list(filter(lambda x : x[1]>=1-delta, zip(pred_x, pred_y))))
false_preds, _ = zip(*list(filter(lambda x : x[1]<delta, zip(pred_x, pred_y))))
unsure_preds, _ = zip(*list(filter(lambda x : delta< x[1]<1-delta, zip(pred_x, pred_y))))

# Unzip the tuples into x and y coordinates
true_x, true_y = zip(*true_preds)
false_x, false_y = zip(*false_preds)
unsure_x, unsure_y = zip(*unsure_preds)

# Plot predictions
plt.scatter(true_x, true_y, color = '#8bc34a', marker = ",")
plt.scatter(false_x, false_y, color = '#D52941', marker = ",")
plt.scatter(unsure_x, unsure_y, color = '#FCD581', marker = ",")
plt.show()


