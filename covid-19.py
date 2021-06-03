import tensorflow as tf
from tensorflow import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
import math

maxValue = 40000
daysToPredict = 7

def preprocessing(values, n):
    result = []
    for i in range(len(values) - n):
        tmp = []
        for j in range(n):
            tmp.append(values[i+j]/maxValue)
        result.append([tmp,values[i+n]/maxValue])
    return result

df = pd.read_excel("data.xlsx", header=None)

# print(df)
dataset = df.values
dataset = dataset[:, 1]

corr = np.correlate(dataset, dataset, mode='full')
corr = corr[int(len(corr)/2):]
print(corr)
most_correlated = np.count_nonzero(corr[0] - np.array(corr) < 10000000000) #17
most_correlated = 14

preprocessed_dataset = preprocessing(dataset, most_correlated)

input_table = []
for i in range(len(preprocessed_dataset) - most_correlated, len(preprocessed_dataset)):
    input_table.append(preprocessed_dataset[i][1])
random.shuffle(preprocessed_dataset)

x_learn = []
y_learn = []
x_test = []
y_test = []

for i in range(len(preprocessed_dataset)):
    if i <= 400:
        x_learn.append(preprocessed_dataset[i][0])
        y_learn.append(preprocessed_dataset[i][1])
    else:
        x_test.append(preprocessed_dataset[i][0])
        y_test.append(preprocessed_dataset[i][1])


x_learn = np.asarray(x_learn).astype(np.float32)
y_learn = np.asarray(y_learn).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

print(x_test, "\n", y_test)

"""
Numbers of neurons in layers:
sqrt(input_nodes * output_nodes)
"""
nodes = int(math.sqrt(most_correlated * daysToPredict))
model = Sequential([
    Dense(32, activation='relu', input_shape=(most_correlated,)),
    Dense(32, activation='relu'),
    Dense(daysToPredict)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

hist = model.fit(x_learn, y_learn,
                 batch_size=32, epochs=100,
                 validation_data=(x_test, y_test))

result = 1 - model.evaluate(x_test, y_test)[1]
print("accuracy: ", result)

predictions = model.predict([input_table])

print("most_correlated: ", most_correlated)
print("input table: ", input_table)
print(predictions)
predictions = [np.asarray(p * maxValue).astype(np.int) for p in predictions]
print(predictions[0])
