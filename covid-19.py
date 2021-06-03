import tensorflow as tf
from tensorflow import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from datetime import datetime
import numpy as np
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

accepted_diff = 0.01
# def linear_regression_equality(y_true, y_pred):
#     diff = K.abs(y_true-y_pred)
#     return K.mean(K.cast(diff < accepted_diff, tf.float32))

def preprocessing(values):
    x = []
    y = []
    for i in range(len(values) - 15):
        tmp = []
        for j in range(15):
            tmp.append(values[i+j]/40000)
        x.append(tmp)
        y.append(values[i+15]/40000)
    return [x, y]


df = pd.read_excel("data.xlsx", header=None)

# print(df)
dataset = df.values
x_learn = dataset[0: 400, 0]
y_learn = dataset[0: 400, 1]
x_test = dataset[401:, 0]
y_test = dataset[401:, 1]

for i in range(len(x_learn)):
    x_learn[i] = int(x_learn[i].timestamp())

for i in range(len(x_test)):
    x_test[i] = int(x_test[i].timestamp())

x_learn = np.asarray(x_learn).astype(np.float32)
y_learn = np.asarray(y_learn).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

res1 = preprocessing(y_learn)
res2 = preprocessing(y_test)

x_learn = res1[0]
y_learn = res1[1]
x_test = res2[0]
y_test = res2[1]

x_learn = np.asarray(x_learn).astype(np.float32)
y_learn = np.asarray(y_learn).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

print(x_test, "\n", y_test)

model = Sequential([
    Dense(32, activation='relu', input_shape=(15,)),
    Dense(32, activation='relu'),
    Dense(1),
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

hist = model.fit(x_learn, y_learn,
                 batch_size=32, epochs=100,
                 validation_data=(x_test, y_test))

result = 1 - model.evaluate(x_test, y_test)[1]
print(result)

tab2 = [0.054175,
        0.027725, 0.04335, 0.0586, 0.05215, 0.041975, 0.0379, 0.026875, 0.013975,
        0.025, 0.031675, 0.03075, 0.02365, 0.019375, 0.014475]
predictions = model.predict([tab2])
print(predictions)
