
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal,rand

'''
# =====================================
grp1 = pd.DataFrame(np.random.randn(20,7), columns=['X','Y','C','D','E','F','G'])
grp1.plot.scatter(x='X', y='Y')  #X,Y열만 선택
print(grp1)
plt.show()
'''
'''
grp2 = pd.Series(np.random.randn(10))
print(grp2)
grp2.plot()
plt.show()
'''
'''
grp3 = pd.Series(np.random.randn(10))
print(grp3)
grp3 = grp3.cumsum() # 각 시점 이전 데이터들의 합
print('====================')
print(grp3)
grp3.plot()
plt.show()
'''

# ======3. 데이터를 시각화하기 대소니
'''
plt.plot([1,3,2,4]) # y축 : 1,3,2,4  x축 : 1,2,3,4(순서대로 자동부여)
plt.plot([1,3,2,7],[10,20,30,40])
#위 두 라인을 같이 표현 가능
# plt.grid(True)
plt.show() #그래프 보임
# ============================
'''
'''
plt.plot([10,20,30,40],[2,6,8,12], label='price')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.legend() #선에 라벨정보를 제공 해 주었다면 범례 표시
plt.title('X Y Graph')
plt.show()
'''

'''
#===========================================
plt.plot([10,20,30,40],[2,6,8,12], label='price')
plt.axis([15, 35, 3, 10])  #X축 범위 :15부터 35  Y축 범위 :3부터 10 범위
plt.legend() #범례 표시
plt.title('X Y Graph')
plt.show()
#=========================================
'''
'''
import numpy as np
d = np.arange(0., 10., 0.4)  #0부터 10까지 0.4 간격으로 배열 생성
plt.plot(d,d*2,'r-', d,d*3,'y--')  #'-'는 실선, --는 점선 (x축, y축, 선스타일,x출,y축,선스타일)

#black 의 k
#red 의 r
#green 의 g
#blue 의 b
#yellow 의 y

plt.xlabel('X value')
plt.ylabel('Y value')
plt.title('X Y Graph')
plt.show()
'''

#막대그래프

'''
s = pd.DataFrame([[1789522, 2655864], [2852440, 4467147]], columns=['a', 'b'],index=['F','M'])
s.plot(kind='bar')
plt.show()
print(s)
'''
