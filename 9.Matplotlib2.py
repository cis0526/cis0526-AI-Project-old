import matplotlib.pyplot as plt

#Matplotlib 색상 지정하기
#포맷 문자열 사용하기
'''
plt.plot([1, 2, 3, 4], [2.0, 2.5, 3.3, 4.5], 'r') #(1,2.0), (2,2.5), (3,3.3), (4,4.5)
plt.plot([1, 2, 3, 4], [2.0, 2.8, 4.3, 6.5], 'g')
plt.plot([1, 2, 3, 4], [2.0, 3.0, 5.0, 10.0], 'b')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

plt.show()
'''
'''

#color 키워드 인자 사용하기

plt.plot([1, 2, 3, 4], [2.0, 2.5, 3.3, 4.5], color='springgreen')
plt.plot([1, 2, 3, 4], [2.0, 2.8, 4.3, 6.5], color='violet')
plt.plot([1, 2, 3, 4], [2.0, 3.0, 5.0, 10.0], color='dodgerblue')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

plt.show()

# Hex code 사용하기

plt.plot([1, 2, 3, 4], [2, 3, 5, 10], color='#e35f62',
         marker='o', linestyle='--')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

plt.show()
'''
'''
#Matplotlib 그래프 영역 채우기
#Matplotlib 그래프의 특정 영역을 색상으로 채워서 강조할 수 있습니다.
#matplotlib.pyplot 모듈에서 그래프의 영역을 채우는 아래의 세가지 함수에 대해 소개합니다.
#fill_between() - 두 수평 방향의 곡선 사이를 채웁니다.
#fill_betweenx() - 두 수직 방향의 곡선 사이를 채웁니다.
#fill() - 다각형 영역을 채웁니다.

#기본 사용
'''
'''
x = [1, 2, 3, 4]
y = [2, 3, 5, 10]

plt.plot(x, y)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.fill_between(x[1:3], y[1:3], alpha=0.5)                   ## fill_between() 사용 alpha: 투명도
#plt.fill_betweenx(y[2:4], x[2:4], color='pink', alpha=0.5)      ## fill_betweenx() 사용

plt.show()
'''

#두 그래프 사이 영역 채우기
'''
x = [1, 2, 3, 4]
y1 = [2, 3, 5, 10]
y2 = [1, 2, 4, 8]

plt.plot(x, y1)
plt.plot(x, y2)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.fill_between(x[1:3], y1[1:3], y2[1:3], color='lightgray', alpha=0.5)

plt.show()
'''

#다각형 영역 채우기
'''
x = [1, 2, 3, 4]
y1 = [2, 3, 5, 10]
y2 = [1, 2, 4, 8]

plt.plot(x, y1)
plt.plot(x, y2)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.fill([1.9, 1.9, 3.1, 3.1], [1.0, 4.0, 6.0, 3.0], color='lightgray', alpha=0.5)

plt.show()
'''
'''
x = [1, 2, 3, 4]
y1 = [2, 3, 5, 10]
y2 = [1, 2, 4, 8]

plt.plot(x, y1)
plt.plot(x, y2)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.fill(x, y1, color='lightgray', alpha=0.5)

plt.show()
'''
'''
fill() 함수에 x, y 값의 리스트를 입력해주면,
각 x, y 점들로 정의되는 다각형 영역을 자유롭게 지정해서 채울 수 있습니다.
'''

#Matplotlib 그리드 설정하기

#기본 사용
'''
import numpy as np

a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
plt.grid(True)

plt.show()
'''

#축 지정하기
'''
import numpy as np

a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='forestgreen', marker='^', markersize=9)
plt.grid(True, axis='y') #{‘both’, ‘x’, ‘y’} 중 선택할 수 있고 디폴트는 ‘both’입니다.

plt.show()
'''

#스타일 설정하기
'''
import numpy as np

a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
plt.grid(True, axis='y', color='red', alpha=0.5, linestyle='--',which='major')

plt.show()
'''
'''
color, alpha, linestyle 파마리터를 사용해서 그리드 선의 스타일을 설정했습니다.
또한 which 파라미터를 ‘major’, ‘minor’, ‘both’ 등으로 사용하면 주눈금, 보조눈금에 각각 그리드를 표시할 수 있습니다.
'''

#Matplotlib 눈금 표시하기
#틱 (Tick)은 그래프의 축에 간격을 구분하기 위해 표시하는 눈금입니다.
# 기본 사용
'''
import numpy as np

a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='forestgreen', marker='^', markersize=9)
plt.xticks([0, 1, 2])
plt.yticks(np.arange(1, 6))

plt.show()
'''

# 눈금 레이블 지정하기
'''
import numpy as np

a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
plt.xticks(np.arange(0, 2.2, 0.2), labels=['Jan', '', 'Feb', '', 'Mar', '', 'May', '', 'June', '', 'July'])
plt.yticks(np.arange(0, 7), ('0', '1GB', '2GB', '3GB', '4GB', '5GB', '6GB'))

plt.show()
'''

#눈금 스타일 설정하기
'''
import matplotlib.pyplot as plt
import numpy as np

a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
plt.xticks(np.arange(0, 2.2, 0.2), labels=['Jan', '', 'Feb', '', 'Mar', '', 'May', '', 'June', '', 'July'])
plt.yticks(np.arange(0, 7), ('0', '1GB', '2GB', '3GB', '4GB', '5GB', '6GB'))

plt.tick_params(axis='x', direction='in', length=3, pad=6, labelsize=14, labelcolor='green', top=True)
plt.tick_params(axis='y', direction='inout', length=10, pad=15, labelsize=12, width=2, color='r')

plt.show()
'''
'''
tick_params() 함수를 사용하면 눈금의 스타일을 다양하게 설정할 수 있습니다.
axis는 설정이 적용될 축을 지정합니다. {‘x’, ‘y’, ‘both’} 중 선택할 수 있습니다.
direction을 ‘in’, ‘out’으로 설정하면 눈금이 안/밖으로 표시됩니다. {‘in’, ‘out’, ‘inout’} 중 선택할 수 있습니다.
length는 눈금의 길이를 지정합니다.
pad는 눈금과 레이블과의 거리를 지정합니다.
labelsize는 레이블의 크기를 지정합니다.
labelcolor는 레이블의 색상을 지정합니다.
top/bottom/left/right를 True/False로 지정하면 눈금이 표시될 위치를 선택할 수 있습니다.
width는 눈금의 너비를 지정합니다.
color는 눈금의 색상을 지정합니다.
'''

#Matplotlib 타이틀 설정하기
# 기본 사용
'''
import numpy as np

a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='forestgreen', marker='^', markersize=9)

plt.tick_params(axis='both', direction='in', length=3, pad=6, labelsize=14)
plt.title('Graph Title')

plt.show()
'''

#위치와 오프셋 지정하기
'''
import numpy as np

a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='forestgreen', marker='^', markersize=9)

plt.tick_params(axis='both', direction='in', length=3, pad=6, labelsize=14)
plt.title('Graph Title', loc='right', pad=20)

plt.show()

{‘left’, ‘center’, ‘right’} 중 선택할 수 있으며 디폴트는 ‘center’입니다.
pad 파라미터는 타이틀과 그래프와의 간격을 포인트 단위로 설정합니다.
'''
'''
#폰트 지정하기

import numpy as np

a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='forestgreen', marker='^', markersize=9)

plt.tick_params(axis='both', direction='in', length=3, pad=6, labelsize=14)
plt.title('Graph Title', loc='right', pad=20)

title_font = {
    'fontsize': 16,
    'fontweight': 'bold'
}
plt.title('Graph Title', fontdict=title_font, loc='left', pad=20)

plt.show()

#fontdict 파라미터에 딕셔너리 형태로 폰트 스타일을 설정할 수 있습니다.
#‘fontsize’를 16으로, ‘fontweight’를 ‘bold’로 설정했습니다.
#‘fontsize’는 포인트 단위의 숫자를 입력하거나 ‘smaller’, ‘x-large’ 등의 상대적인 설정을 할 수 있습니다.
#‘fontweight’에는 {‘normal’, ‘bold’, ‘heavy’, ‘light’, ‘ultrabold’, ‘ultralight’}와 같이 설정할 수 있습니다.
'''
'''
# 타이틀 얻기
import numpy as np

a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='forestgreen', marker='^', markersize=9)

plt.tick_params(axis='both', direction='in', length=3, pad=6, labelsize=14)
title_right = plt.title('Graph Title', loc='right', pad=20)

title_font = {
    'fontsize': 16,
    'fontweight': 'bold'
}
title_left = plt.title('Graph-Title', fontdict=title_font, loc='left', pad=20)

print(title_left.get_position())
print(title_left.get_text())

print(title_right.get_position())
print(title_right.get_text())

plt.show()
'''
'''
#Matplotlib 수직선/수평선 표시하기
axhline(): 축을 따라 수평선을 표시합니다.
axvline(): 축을 따라 수직선을 표시합니다.
hlines(): 지정한 점을 따라 수평선을 표시합니다.
vlines(): 지정한 점을 따라 수직선을 표시합니다.
#기본 사용 (axhline/axvline)
'''
'''
import numpy as np

a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='forestgreen', marker='^', markersize=9)
plt.xticks(np.arange(0, 2.2, 0.2), labels=['Jan', '', 'Feb', '', 'Mar', '', 'May', '', 'June', '', 'July'])
plt.yticks(np.arange(0, 7), ('0', '1GB', '2GB', '3GB', '4GB', '5GB', '6GB'))

plt.axhline(1, 0, 0.55, color='gray', linestyle='--', linewidth='1')
plt.axvline(1, 0, 0.50, color='lightgray', linestyle=':', linewidth='2')

plt.axhline(5.83, 0, 0.95, color='gray', linestyle='--', linewidth='1')
plt.axvline(1.8, 0, 0.95, color='lightgray', linestyle=':', linewidth='2')

plt.show()
'''
'''
axhline() 함수의 첫번째 인자는 y 값으로서 수평선의 위치가 됩니다.
두, 세번째 인자는 xmin, xmax 값으로서 0에서 1 사이의 값이며, 0은 왼쪽 끝, 1은 오른쪽 끝을 의미합니다.
axvline() 함수의 첫번째 인자는 x 값으로서 수직선의 위치가 됩니다.
두, 세번째 인자는 ymin, ymax 값으로서 0에서 1 사이의 값을 입력합니다.
0은 아래쪽 끝 (X축), 1은 위쪽 끝을 의미합니다.

#기본 사용 (hlines/vlines)
'''
'''
import matplotlib.pyplot as plt
import numpy as np

a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='forestgreen', marker='^', markersize=9)
plt.xticks(np.arange(0, 2.2, 0.2), labels=['Jan', '', 'Feb', '', 'Mar', '', 'May', '', 'June', '', 'July'])
plt.yticks(np.arange(0, 7), ('0', '1GB', '2GB', '3GB', '4GB', '5GB', '6GB'))

plt.hlines(4, 1, 1.6, colors='pink', linewidth=3)
plt.vlines(1, 1, 4, colors='pink', linewidth=3)

plt.show()
'''
'''
hlines() 함수에 y, xmin, xmax를 순서대로 입력하면, 점 (xmin, y)에서 점 (xmax, y)를 따라 수평선을 표시합니다.
vlines() 함수에 x, ymin, ymax를 순서대로 입력하면, 점 (x, ymin)에서 점 (x, ymax)를 따라 수평선을 표시합니다.
'''
