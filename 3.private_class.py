import numpy as np
import keras as k
#공부시간 X와 성적 Y의 리스트를 만듭니다.
x = np.array([[2, 0], [4, 4], [6, 2], [8, 3]])
y = np.array([[81], [93], [91], [97]])

np.random.seed(3)
model = k.models.Sequential()
model.add(k.layers.Dense(2, input_dim=2))
model.add(k.layers.Dense(4))
model.add(k.layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

# 훈련
model.fit(x, y, epochs=6010) # 100번 반복 훈련

# 테스트
a=int(input("stutime?"))
b=int(input("private?"))
print(model.predict([[a,b]]))  # 입력이 5일 때 결과값 출력

