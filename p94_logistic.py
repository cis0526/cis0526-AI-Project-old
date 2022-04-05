import numpy as np
import keras as k

# 훈련용 데이터
data = [[2, 0], [4, 0], [6, 0], [8, 1],[10,1],[12,1],[14,1]]
x = [i[0] for i in data]
y = [i[1] for i in data]

#리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸어 줍니다.(인덱스를 주어 하나씩 불러와 계산이 가능해 지도록 하기 위함입니다.)
x_data = np.array(x)
y_data = np.array(y)

print(x)


np.random.seed(3)
# 모델 생성
model = k.models.Sequential()
# 입력 값은 1차원 자료: input_dim=1
model.add(k.layers.Dense(1, input_dim=1,activation='sigmoid'))
#model.add(k.layers.Dense(4))
#model.add(k.layers.Dense(1))

# 최적화 방식

# cost/loss 함수로 컴파일 작업
#adam=k.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
  #metrics=['accuracy']  선형회귀모델에서는 사용 불가
# 훈련
model.fit(x_data, y_data, epochs=100,verbose=1)  # 100번 반복 훈련 verbose : 훈련정보

# 테스트
print(model.predict(np.array([8])))  # 입력이 5일 때 결과값 출력