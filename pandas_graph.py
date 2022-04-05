import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#matplotlib 구성
#https://wikidocs.net/4763
#matplotlib에서 그래프는 Figure 객체 내에 존재합니다. 따라서 그래프를 그리려면 다음 코드에서처럼 figure 함수를 사용해 Figure
# 객체를 생성해야 합니다.
'''
fig = plt.figure()
fig.add_subplot(1, 1, 1)  #빈 Figure 객체에 Axes 객체(또는 subplot)를 생성하려면 add_subplot 메서드를 사용하면 됩니다.
#type(fig)
plt.show() #Figure 객체를 생성한 후 plt.show 함수를 호출하면   Figure 객체가 화면에 출력됩니다.
           #Axes 객체가 아직 포함되지 않으면  그래프를 그릴 수는 없습니다. 비어있는 객체 출력
'''

# =====================================
'''
grp1 = pd.DataFrame(np.random.randn(20,7), columns=['X','Y','C','D','E','F','G'])
grp1.plot.scatter(x='X', y='Y')  #X,Y열만 선택
print(grp1)
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
'''
grp4 = pd.Series(np.random.randn(10))
grp4.plot.kde()  #
print(grp4)
# ======3. 데이터를 시각화하기 대소니
plt.show()
plt.plot([1,2,3,4])

# plt.plot([1,3,2,7],[10,20,30,40])
# plt.grid(True)
# plt.show() #그래프 보임
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
'''
d = np.arange(0., 10., 0.4)

plt.figure(1)

plt.subplot(211) #2 : 가로방향 2, 세로방향 1, 그리고 마지막 숫자는 몇번 째 위치??
plt.plot(d,d*2,'r-')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.title('Double Graph')

plt.subplot(212)  # 마지막 숫자를 상호 교환 해 볼것
plt.plot(d,d*-2,'b--')
plt.xlabel('X value')
plt.ylabel('Y value')

plt.show()
'''
# =============================
'''
plt.scatter(np.random.normal(5, 3, 1000), np.random.normal(3, 5, 1000)) #(x축,y축)
plt.show()
'''

#막대그래프
#https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html  참조
'''
s = pd.DataFrame([[1789522, 2655864], [2852440, 4467147]], columns=['a', 'b'],index=['F','M'])
s.plot(kind='bar')
plt.show()
print(s)
'''
'''
s = pd.DataFrame([[1789522, 2655864], [2852440, 4467147]], columns=['a', 'b'],index=['F','M'])
#s.plot(kind='bar')
s.plot(kind='bar', title="Year", rot=0, color='blue') #색깔 조견표 https://codetorial.net/matplotlib/set_color.html
plt.show()
'''
'''
x = np.arange(3)
years = ['2017', '2018', '2019']
values = [100, 400, 900]

plt.bar(x, values, width=0.6, align='edge', color="springgreen",
        edgecolor="gray", linewidth=3, tick_label=years, log=True)
'''
#width: 막대의 너비입니다. 디폴트 값은 0.8이며, 0.6으로 설정
#align : 틱 (tick)과 막대의 위치를 조절합니다. 디폴트 값은 ‘center’인데, ‘edge’로 설정하면 막대의 왼쪽 끝에 x_tick이 표시
#color : 막대의 색 지정.
#edgecolor : 막대의 테두리 색 지정.
#linewidth : 테두리의 두께 지정
#tick_label :  어레이 형태로 지정하면, 틱에 어레이의 문자열을 순서대로 나타낼 수 있다.
#log=True로 설정하면, y축이 로그 스케일로 표시
#plt.show()
#===================================================
#subplot
#plt.subplot(2,1,1)
#plt.subplot(2,1,2)
#plt.show()

# subplots의 예
'''
fig, axes = plt.subplots(nrows=2, ncols=1)
print (fig,axes)
plt.show()
'''
#fig란 figure로써 - 전체 subplot을 말한다. ex) 서브플로안에 몇개의 그래프가 있던지 상관없이  그걸 담는 하나 전체 사이즈를 말한다.
#ax는 axe로써 - 전체 중 낱낱개를 말한다 ex) 서브플롯 안에 2개(a1,a2)의 그래프가 있다면 a1, a2 를 일컬음
#================================================

'''
car = ('BMW', 'BENZ', 'KIA', 'HD')
x_pos = np.arange(len(car)) #len(car)=4
speed = 80 + 60 * np.random.rand(len(car)) #난수 네개 생성/speed는 리스트다입  네개 데이터 보유

#plt.subplots()은 Figure와 Axes 객체가 들어있는 Tuple을 반환하는 함수입니다.
# 따라서 fig, ax = plt.subplots()을 사용할 때이 튜플을 fig 및 ax 변수로 압축합니다.
# fig을 사용하면 그림 수준 속성을 변경하거나 그림을 나중에 이미지 파일로 저장하려는 경우
# (예 : fig.savefig('yourfilename.png') 사용) 유용합니다.
# 반환 된 Figure 객체를 사용할 필요는 없지만 많은 사람들이 나중에 사용하므로 일반적으로 볼 수 있습니다.
fig, ax = plt.subplots()
print(fig,ax)
ax.bar(x_pos, speed, align='center', color='blue')
ax.set_xticks(x_pos)
ax.set_xticklabels(car)
plt.show()
'''
'''
#====================================
#캔들차트
import matplotlib.ticker as ticker
from mplfinance.original_flavor import candlestick2_ohlc
from matplotlib import font_manager, rc
# 한글 폰트 지정
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
fig, ax = plt.subplots() # x-축 날짜
xdate = df.날짜.astype('str')
for i in range(len(xdate)):
        xdate[i] = xdate[i][2:] # 2020-01-01 => 20-01-01
# 종가 및 5,20,60,120일 이동평균
ax.plot(xdate, df['종가'], label="종가",linewidth=0.7,color='k')
ax.plot(xdate, df['종가'].rolling(window=5).mean(), label="평균5일",linewidth=0.7)
ax.plot(xdate, df['종가'].rolling(window=20).mean(), label="평균20일",linewidth=0.7)
ax.plot(xdate, df['종가'].rolling(window=60).mean(), label="평균60일",linewidth=0.7)
ax.plot(xdate, df['종가'].rolling(window=120).mean(), label="평균120일",linewidth=0.7)
candlestick2_ohlc(ax,DataFrame['시가'],df['고가'],df['저가'],df['종가'], width=0.5, colorup='r', colordown='b')
fig.suptitle("캔들 스택 차트 예시")
ax.set_xlabel("날짜")
ax.set_ylabel("주가(원)")
ax.xaxis.set_major_locator(ticker.MaxNLocator(25)) # x-축에 보일 ticker 개수 ~20개이면 1달
ax.legend(loc=1) # legend 위치
plt.xticks(rotation = 45) # x-축 글씨 45도 회전
plt.grid() # 그리드 표시
plt.show()
'''
