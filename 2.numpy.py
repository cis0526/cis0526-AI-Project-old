import numpy as np

'''
x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)  # type : np.int32, np.float64
print(x)
x = x.astype(np.float64) #astype : type 바꾸기
print(x)

'''
# 연산

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[2, 3], [4, 5]])
#print (arr1)


#num=(arr1.max())   #최대 값
# num=(arr1.min())   #최소 값
# num=(arr1.mean())  #평균 값
# num=(arr1.sum())   #전체 합
# print (num)

# .reshape
'''
arr3=arr1.reshape(4)  #2차원을 1차원으로, 원소 갯수가  동일해야 함
print (arr3)
arr4=arr3.reshape(2,2) #1차원을 2차원으로  (차원, 원소갯수)
arr5=arr3.reshape(2,2,order="f") #포트란 타입
print (arr4)
print (arr5)
'''

'''
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False)
start :시작 값
stop : 끝 값 
num : 생성 할 요소의 갯수
endpoin : True인 경우 마지막을 요소 값으로 끝값 포함.
retstep : True이면 step값을 반환
'''

x = np.linspace(1.0, 100.0, num=100,retstep=False)  # (초기값, 최종값, num=원소 갯수)
#x = np.linspace(1.0, 100.0, num=100,endpoint=False,retstep=False)
#x = np.linspace(1.0, 100.0, num=100,retstep=True)
print(x)
x1=x.reshape(4,25,order="C") #,retstep=True 인 경우에는 반환값 때문에 리스트 구조 때문에 위배 됨,  출력 확인 권유
x2=x.reshape(4,25,order="f")
#x1 = x.reshape(4, 5, 5, order="C") #3차원
#x1=x.reshape(4,-1)  # 1차원은 4로 정하고 2차원은 자동으로 25
#print(x)
#print(x1)

#print(x1, x2)

'''
order는
C: C형식 Index 순서, 뒤 차원부터 변경하고 앞 쪽 차원을 변경
(기본 값이 C이므로 별도 지정하지 않으면 모두 C이므로 마음 놓고 사용해도 된다) 
F: 포트란형식 Index 순서, 앞 차원부터 변경하고 뒷 쪽 차원을 변경
'''
'''
x2 = x1.reshape(4, 25, order="C")
print(np.array_equal(x, x2))  #배열 x와 x2가 같은지 여부
print(x1, x2)
'''
# x=np.linspace(1.0,100.0, num=100,endpoint=True)  #endpoint=False는 최종값 미포함(True는 포함)
# x=np.linspace(1.0,100.0, num=100,retstep=True)   #step값 반환

# print (x)
