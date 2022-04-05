#라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#seaborn에서 제공하는 flights 데이터 셋을 사용
flights = sns.load_dataset('flights')
print(flights)

#그래프 사이즈 설정
plt.figure(figsize=(12, 3))

sns.barplot(data=flights, x="year", y="passengers")
plt.figure(figsize=(12, 3))
sns.violinplot(data=flights, x="year", y="passengers")
plt.figure(figsize=(12, 3))
sns.swarmplot(data=flights, x="year", y="passengers")
plt.figure(figsize=(12, 3))
sns.lineplot(data=flights, x="year", y="passengers")
plt.show()

