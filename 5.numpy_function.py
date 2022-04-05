# Numpy 함수

import numpy as np
from numpy.core._multiarray_umath import ndarray

#arr1 = np.random.randn(5, 3)  # 가우스 난수
#print(arr1)
print('---------------')
# print(np.abs(arr1))  #절대 값
# print(np.sqrt(arr1)) # 제곱근 음수인 경우 non 출력
# print(np.square(arr1)) # 각 요소의 거듭제곱 반환
# print(np.sign(arr1))  # 각 요소의 부호 (+인 경우 1, -인 경우 -1, 0인 경우 0)
# print(np.ceil(arr1))  #소수 첫 번째 자리 이하에서 올림한 값을 계산하기
# print(np.floor(arr1))  # 소수 첫 번째 자리에서 내림한 값을 계산하기
# print(np.isnan(arr1))   #NaN인 경우 True를, 아닌 경우 False를 반환하기
#print(np.isinf(arr1))  #무한대인 경우 True를, 아닌 경우 False를 반환하기
# print(np.cos(arr1))  # 삼각함수 값을 계산하기(cos, cosh, sin, sinh, tan, tanh)
# print(np.sort(arr1)) # 각 행의 오름차순 정렬
# print(np.sort(arr1, axis=0))   #열을 오름차순으로
# print(np.sort(arr1, axis=1))   #행을 오름차순으로
print('---------------')
#arr1 = np.array([5,6,4,8,2,9])
#print(np.sort([5,6,4,8,2,9]))
#print(np.sort(arr1)[::-1])
#print(np.sort([[5,6,4,8,2,9],[7,3,5,9,2,8]],axis=1)[::-1]) # -1:내림차순 1:오름차순, 2차원에서는 사용 안함
# [::-1]  :  역순  배열 뒤집음
#arr2 = np.random.randn(5, 3)
#print(arr2)
#print('---------------')


arr1=np.array([[1,2],[3,4],[5,6]])
arr2=np.array([[7,8],[9,10],[11,12]])
#arr2=np.array([[1,2,3],[4,5,6]])

print(np.multiply(arr1,arr2))  #두 개의 array에 대해 동일한 위치의 성분끼리 연산 값을 계산하기(add, subtract, multiply, divide)
#print(np.matmul(arr1,arr2))  #행렬  매트릭스 곱
#print(np.maximum(arr1,arr2))   # 두 개의 array에 대해 동일한 위치의 성분끼리 비교하여 최대값 또는 최소값 계산하기(maximum, minimum)

