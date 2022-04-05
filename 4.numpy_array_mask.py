import numpy as np
from numpy.core._multiarray_umath import ndarray

names = np.array(['choi', 'choi', 'kim', 'kang', 'Lee', 'kim', 'Park', 'kang'])
names_mask = (names == 'kim')  # name 요소 중에서 'kim' 은 True, 그 외는 False값  반환
#print(names)
#print(names_mask)

data = np.random.rand(8, 4)
print(data)
print('-----------------------------------')
#print(data[names_mask, :])  #names_mask값 중에서 true값에 해당하는 2,5행 출력
#print(data[names == 'kim',:])  #names 요소가 'kim'인 경우의 위치 값에 위치한 data의 행 요소 반환 ':'은 의미 없음
#print(data[(names == 'kim') | (names == 'kang'),:])  #OR

#print(data[:,0] < 0.5) #0번째 열이 0.5보다 작은 요소의 boolean 값을 반환
#print(data[data[:,0]<0.5,:]) # 0번째 열의 값이 0.5보다 작은 행을 구한다
#print(data[data[:,0]<0.5,2:4]) #번째 열의 값이 0.5보다 작은 행의 2,3번째 열 값
#data[data[:,0]<0.5,2:4] = 0 #번째 열의 값이 0.5보다 작은 행의 2,3번째 열 값을 0으로 치환
#print(data)
