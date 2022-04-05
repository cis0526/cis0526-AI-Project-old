'''
matplotlib.pyplot 모듈은 MATLAB과 비슷하게 명령어 스타일로 동작하는 함수들의 모음입니다.
matplotlib.pyplot 모듈의 각각의 함수를 사용해서 간편하게 그래프를 만들고 변화를 줄 수 있습니다.
예를 들어, 그래프 영역을 만들고, 몇 개의 선을 표현하고, 레이블로 꾸미는 등의 작업을 할 수 있습니다.
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#기본 그래프 그리기
#점찍기
'''
plt.scatter([1,2,3,4], [1,2,3,4])
plt.show()
'''
'''
plt.plot([1, 2, 3, 4])
plt.show()
'''
'''
#matplotlib.pyplot 모듈의 plot() 함수에 하나의 숫자 리스트를 입력함으로써 아래와 같은 그래프가 그려집니다.
#plot() 함수는 리스트의 값들이 y 값들이라고 가정하고, x 값들 ([0, 1, 2, 3])을 자동으로 만들어냅니다.
#x 값은 기본적으로 [0, 1, 2, 3]이 되어서, 점 (0, 1), (1, 2), (2, 3), (3, 4)를 잇는 아래와 같은 꺾은선 그래프가 나타납니다.
'''
'''
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()
'''
'''

#예를 들어, 위와 같이 두 리스트를 입력하면, x와 y 값을 그래프로 나타낼 수 있습니다.


# 스타일 지정하기
#x, y 값 인자에 대해 선의 색상과 형태를 지정하는 포맷 문자열을 세 번째 인자에 입력할 수 있습니다.

'''
'''
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'g--')
plt.axis([0, 6, 0, 20]) #축의 범위 ([xmin, xmax, ymin, ymax])
plt.show()
'''

#형식 문자열은 ‘ro’는 빨간색 (‘red’)의 원형 (‘o’) 마커를 의미합니다.
#또한, 예를 들어 ‘b-‘는 파란색 (‘blue’)의 실선 (‘-‘)을 의미합니다. ('--')는 점선 x,v,^
# alpha 속성: 선의 투명도 (0.0 ~ 1.0)

#여러 개의 그래프 그리기
'''
import numpy as np

t = np.arange(0., 5., 0.2)  # 200ms 간격으로 균일한 샘플된 시간

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()
'''
'''
#Matplotlib에서는 일반적으로 NumPy 어레이를 이용하게 되는데,
#사실 NumPy 어레이를 사용하지 않더라도 모든 시퀀스는 내부적으로 NumPy 어레이로 변환됩니다.
'''
'''
plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis',rotation='horizontal',color='r'  ) #각각의 축에 레이블 표시
plt.show()
'''
'''
Property     Description        Option
================================================================
alpha        텍스트의 투명도    0.0 ~ 1.0 (float)
color        텍스트의 색상      Any Matplotlib color
family       텍스트의 글꼴     [‘serif’ | ‘sans-serif’ | ‘cursive’ | ‘fantasy’ | ‘monospace’ ]
rotation     텍스트의 회전각    [‘vertical’ | ‘horizontal’ ]
size         텍스트의 크기  [‘xx-small’ | ‘x-small’ | ‘small’ | ‘medium’ | ‘large’ | ‘x-large’ | ‘xx-large’ ]
weight       텍스트의 굵기 [‘ultralight’ | ‘light’ | ‘normal’ | ‘regular’ | ‘book’ | ‘medium’ | ‘roman’ | ‘semibold’
                          | ‘demibold’ | ‘demi’ | ‘bold’ | ‘heavy’ | ‘extra bold’ | ‘black’ ]
'''
'''
#여백 지정하기

plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
plt.xlabel('X-Axis', labelpad=10)  #축과 레이블의 여백 (Padding)을 지정
plt.ylabel('Y-Axis', labelpad=10)
plt.show()
'''
'''
#폰트 설정하기
plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
plt.xlabel('X-Axis', labelpad=15, fontdict={'family': 'serif', 'color': 'b', 'weight': 'bold', 'size': 14})
plt.ylabel('Y-Axis', labelpad=20, fontdict={'family': 'fantasy', 'color': 'deeppink', 'weight': 'normal', 'size': 'xx-large'})
plt.show()
'''
'''
#아래와 같이 작성하면 폰트 스타일을 편리하게 재사용할 수 있습니다.

font1 = {'family': 'serif',
         'color': 'b',
         'weight': 'bold',
         'size': 14
         }

font2 = {'family': 'fantasy',
         'color': 'deeppink',
         'weight': 'normal',
         'size': 'xx-large'
         }

plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
plt.xlabel('X-Axis', labelpad=15, fontdict=font1)
plt.ylabel('Y-Axis', labelpad=20, fontdict=font2)
plt.show()
'''
'''
#범례 지정 
#plot() 함수에 label 문자열을 지정하고, matplotlib.pyplot 모듈의 legend() 함수를 호출
plt.plot([1, 2, 3, 4], [2, 3, 5, 10], label='Price ($)')
#plt.plot([1, 2, 13, 14], [12, 13, 5, 10], label='Price2 ($)')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.legend() #legend() (레전드)

plt.show()
'''
'''
#위치 지정하기
plt.plot([1, 2, 3, 4], [2, 3, 5, 10], label='Price ($)')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
#plt.legend(loc=(0.0, 0.0)) #loc=(0.0, 0.0)은 데이터 영역의 왼쪽 아래, loc=(1.0, 1.0)은 데이터 영역의 오른쪽 위 위치
plt.legend(loc=(0.5, 0.5))
#plt.legend(loc=(1.0, 1.0))
plt.show()
'''
'''

plt.plot([1, 2, 3, 4], [2, 3, 5, 10], label='Price ($)')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.legend(loc='lower right') #loc=’lower right’와 같이 지정하면 아래와 같이 오른쪽 아래에 범례가 표시됩니다.
#plt.legend(loc=4)
plt.show()
'''
'''
Location String     Location Code                            설명
====================================================================================================

‘best’                       0             그래프의 최적의 위치에 표시합니다. (디폴트)
‘upper right’                1             그래프의 오른쪽 위에 표시합니다.
‘upper left’                 2             그래프의 왼쪽 위에 표시합니다.
‘lower left’                 3             그래프의 왼쪽 아래에 표시합니다.
‘lower right’                4             그래프의 오른쪽 아래에 표시합니다.
‘right’                      5             그래프의 오른쪽에 표시합니다.
‘center left’                6             그래프의 왼쪽 가운데에 표시합니다.
‘center right’               7             그래프의 오른쪽 가운데에 표시합니다.
‘lower center’               8             그래프의 가운데 아래에 표시합니다.
‘upper center’               9             그래프의 가운데 위에 표시합니다.
‘center’                    10             그래프의 가운데에 표시합니다.
'''
'''
#열 개수 지정하기

plt.plot([1, 2, 3, 4], [2, 3, 5, 10], label='Price ($)')
plt.plot([1, 2, 3, 4], [3, 5, 9, 7], label='Demand (#)')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
# plt.legend(loc='best')          # ncol = 1
plt.legend(loc='best', ncol=2)    # ncol = 2

plt.show()
'''
'''
#폰트 크기 지정하기

plt.plot([1, 2, 3, 4], [2, 3, 5, 10], label='Price ($)')
plt.plot([1, 2, 3, 4], [3, 5, 9, 7], label='Demand (#)')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
# plt.legend(loc='best')
plt.legend(loc='best', ncol=2, fontsize=14)

plt.show()
'''
'''
#범례 테두리 꾸미기

plt.plot([1, 2, 3, 4], [2, 3, 5, 10], label='Price ($)')
plt.plot([1, 2, 3, 4], [3, 5, 9, 7], label='Demand (#)')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
# plt.legend(loc='best')
plt.legend(loc='best', ncol=2, fontsize=14, frameon=True, shadow=True)

#frameon 파라미터는 범례 텍스트 상자의 테두리를 표시할지 여부를 지정합니다.
#shadow 파라미터를 사용해서 텍스트 상자에 그림자를 표시할 수 있습니다.

plt.show()
'''
#legend() 함수에는 facecolor, edgecolor, borderpad, labelspacing과 같은 다양한 파라미터가 있습니다.
'''
matplotlib.pyplot 모듈의 xlim(), ylim(), axis() 함수를 사용하면 그래프의 X, Y축이 표시되는 범위를 지정할 수 있습니다.
xlim() - X축이 표시되는 범위를 지정하거나 반환합니다.
ylim() - Y축이 표시되는 범위를 지정하거나 반환합니다.
axis() - X, Y축이 표시되는 범위를 지정하거나 반환합니다.
'''
'''

plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
# plt.xlim([0, 5])          # X축의 범위: [xmin, xmax]
# plt.ylim([0, 20])         # Y축의 범위: [ymin, ymax]
plt.axis([0, 5, 0, 20])     # X, Y축의 범위: [xmin, xmax, ymin, ymax]
#print(plt.ylim())
plt.show()
'''
'''
#옵션 지정하기

plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
# plt.axis('square')
plt.axis('tight')

plt.show()
'''
'''
axis() 함수는 아래와 같이 축에 관한 다양한 옵션을 제공합니다.
값              설명 
=============================================================
'on'        축과 라벨을 켠다.
'off'       축과 라벨을 끈다.
'equal'     각 축의 범위와 축의 스케일을 동일하게 설정한다.
'scaled'    플롯 박스의 차원과 동일하게 축의 스케일을 설정한다.
'tight'     모든 데이터를 볼 수 있을 정도로 축의 범위를 충분히 크게 설정한다.
'auto'      축의 스케일을 자동으로 설정한다.
'normal'    'auto'와 동일하다.
'image'      데이터 범위에 대해 축의 범위를 사용한 'scaled'이다.
'square'     각축의 범위 즉 xmax-xmin=ymax-ymin 되도록 설정한다.
'''
#축 범위 얻기
'''
plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

x_range, y_range = plt.xlim(), plt.ylim() #X축, Y축의 범위 반환
print(x_range, y_range)

axis_range = plt.axis('scaled')
print(axis_range) #axis() 함수는 그래프 영역에 표시되는 X, Y축의 범위를 반환합니다.
plt.show()
'''
'''

plt.plot([1, 2, 3], [4, 4, 4], '-', color='C0', label='Solid')
plt.plot([1, 2, 3], [3, 3, 3], '--', color='C0', label='Dashed')
plt.plot([1, 2, 3], [2, 2, 2], ':', color='C0', label='Dotted')
plt.plot([1, 2, 3], [1, 1, 1], '-.', color='C0', label='Dash-dot')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.axis([0.8, 3.2, 0.5, 5.0]) #x축 y축의 범위
plt.legend(loc='upper right', ncol=4) #범례 위치 및 갯수
plt.show()
'''
'''
Matplotlib에서 선의 종류를 지정하는 가장 간단한 방법은 포맷 문자열을 사용하는 것입니다.
‘ - ‘ (Solid), ‘ - - ‘ (Dashed), ‘ : ‘ (Dotted), ‘ -. ‘ (Dash-dot)과 같이 네가지 종류를 선택할 수 있습니다.
'''
#linestyle 지정하기
'''

plt.plot([1, 2, 3], [4, 4, 4], linestyle='solid', color='C0', label="'solid'")
plt.plot([1, 2, 3], [3, 3, 3], linestyle='dashed', color='C0', label="'dashed'")
plt.plot([1, 2, 3], [2, 2, 2], linestyle='dotted', color='C0', label="'dotted'")
plt.plot([1, 2, 3], [1, 1, 1], linestyle='dashdot', color='C0', label="'dashdot'")
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.axis([0.8, 3.2, 0.5, 5.0])
plt.legend(loc='upper right', ncol=4)
plt.tight_layout()
plt.show()
'''

#plot() 함수의 linestyle 파라미터 값을 직접 지정할 수 있습니다.

'''
#튜플 사용하기

plt.plot([1, 2, 3], [4, 4, 4], linestyle=(0, (1, 1)), color='C0', label='(0, (1, 1))')
plt.plot([1, 2, 3], [3, 3, 3], linestyle=(0, (1, 5)), color='C0', label='(0, (1, 5))')
plt.plot([1, 2, 3], [2, 2, 2], linestyle=(0, (5, 1)), color='C0', label='(0, (5, 1))')
plt.plot([1, 2, 3], [1, 1, 1], linestyle=(0, (3, 5, 1, 5)), color='C0', label='(0, (3, 5, 1, 5))')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.axis([0.8, 3.2, 0.5, 5.0])
plt.legend(loc='upper right', ncol=2)
plt.tight_layout()
plt.show()

#linestyle=(0, (5, 0)) (선의시작위치,(선의 길이, 공백의 간격))) 선의 시작위치 값이 커지면  실제 보이는 첫번 쩨 선의 길이는  짧아진다.
                       #보이는 위치보다 앞 쪽으로 당겨 지기 때문이다.
'''
'''
#특별한 설정이 없으면 그래프가 실선으로 그려지지만, 위의 그림과 같은 마커 형태의 그래프를 그릴 수 있습니다.
#plot() 함수의 포맷 문자열 (Format string)을 사용해서 그래프의 선과 마커를 지정하는 방법에 대해 알아봅니다.

#기본 사용
'''
'''
plt.plot([1, 2, 3, 4], [2, 3, 5, 10], 'bo')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()
'''

#plot() 함수에 숫자 데이터와 함께 ‘bo’를 입력해주면 파란색의 원형 마커로 그래프가 표시됩니다.
#‘b’는 blue, ‘o’는 circle을 나타내는 문자입니다.

#선/마커 동시에 나타내기
'''
# plt.plot([1, 2, 3, 4], [2, 3, 5, 10], 'bo-')    # 파란색 + 실선 + 마커
plt.plot([1, 2, 3, 4], [2, 3, 5, 10], 'bo--')     # 파란색 + 점선 + 마커
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()
'''

#선/마커 표시 형식
#C:\Users\A310-01\Desktop\강의자료\2021\인공지능1 의 이미지 참조
