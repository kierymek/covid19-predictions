import tensorflow as tf
from tensorflow import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
import math
import matplotlib.pyplot as plt

maxValue = 40000
daysToPredict = 7


def preprocessing(values, x_number, y_number):
    result = []
    for i in range(len(values) - x_number - y_number):
        tmp = []
        tmp2 = []
        for j in range(x_number):
            tmp.append(values[i+j]/maxValue)
        for k in range(y_number):
            tmp2.append(values[i+x_number+k]/maxValue)

        result.append([tmp, tmp2])
    return result


df = pd.read_excel("data.xlsx", header=None)

dataset = df.values
dataset = dataset[:, 1]

corr = np.correlate(dataset, dataset, mode='full')
corr = corr[int(len(corr)/2):]
print(corr)
most_correlated = np.count_nonzero(
    corr[0] - np.array(corr) < 10000000000)  # 17
most_correlated = 14

plt.plot(corr)
plt.title('Autocorrelation')
plt.xlabel('Next offsets')
plt.ylabel('Autocorrelation')
plt.show()

preprocessed_dataset = preprocessing(dataset, most_correlated, daysToPredict)

input_table = []
for i in range(len(preprocessed_dataset) - most_correlated, len(preprocessed_dataset)):
    input_table.append((preprocessed_dataset[i][1])[0])
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

x_plot = x_learn[:5]
y_plot = y_learn[:5]

x_learn = np.asarray(x_learn).astype(np.float32)
y_learn = np.asarray(y_learn).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

model = Sequential([
    Dense(32, activation='relu', input_shape=(most_correlated,)),
    Dense(32, activation='relu'),
    Dense(daysToPredict)
])

model.compile(optimizer='adam', loss='mean_squared_error',
              metrics=['mean_squared_error'])

hist = model.fit(x_learn, y_learn,
                 batch_size=32, epochs=100,
                 validation_data=(x_test, y_test))

result = 1 - model.evaluate(x_test, y_test)[1]
print("accuracy: ", result)

# predictions for next 7 days using last 14 records
predictions = model.predict([input_table])

plot_predictions = []

for i in range(len(x_plot)):
    plot_predictions.append(
        [(p * maxValue).astype(np.int) for p in model.predict([x_plot[i]])])

new_y_plot = []
for i in range(len(y_plot)):
    new_y_plot.append([int(p * maxValue) for p in y_plot[i]])

y_plot = new_y_plot

print("plot_predictions: ", plot_predictions)
print("y_plot: ", y_plot)


print("most_correlated: ", most_correlated)
print("input table: ", input_table)
print("predictions: ", predictions)
predictions = [(p * maxValue).astype(np.int) for p in predictions]
print(predictions[0])



for i in range(len(plot_predictions)):
    plt.plot(plot_predictions[i][0], color='red', label="Prediction")

    plt.plot(y_plot[i], label="Reality")
    plt.title('Infections')
    plt.xlabel('Next days')
    plt.ylabel('Infections per day')
    plt.legend()
    plt.show()

    err = []
    for j in range(0, 7):
        err.append(math.fabs(y_plot[i][j] - plot_predictions[i][0][j]) / maxValue * 100)

    plt.plot(err, color='green', linestyle='dashed', linewidth=3,
         marker='o', markerfacecolor='blue', markersize=12)
    plt.title('Error')
    plt.xlabel('Next days')
    plt.ylim(0, 4)
    plt.ylabel('Error in percents [%]')
    plt.show()
