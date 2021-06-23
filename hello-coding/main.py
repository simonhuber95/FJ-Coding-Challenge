from datagen import genererate_data

# Example how to draw data from distribution
true_data, false_data  = genererate_data(10)
print(true_data, false_data)

# ToDO: Write a Classifier in Keras and train it on the data set.
# Try for an accuracy of >95% in the test set