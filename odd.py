
#[Case B]
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import np_utils

X = np.array([[0], [1], [2], [3], [5], [6], [8], [9],[10], [11], [12], [13], [14], [15], [16], [17], [18], [19]])
Y = np.array([[0], [1], [0], [1], [1], [0], [0], [1],[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]])
#X = np_utils.to_categorical(Y)

model = Sequential()
model.add(Dense(16, input_dim=1, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])

model.fit(X, Y, epochs=60000)
n=10
while (n!=0) :
#    n=int(input("숫자 입력"))
#    X_hat = np.array([n])
    Y_hat = model.predict([int(input("숫자 입력"))])
    print (Y_hat)
    if Y_hat>0.5 :
        print("홀수")
    else :
        print("짝수")
