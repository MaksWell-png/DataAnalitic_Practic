import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Загрузка датасета из CSV-файла
data = pd.read_csv('FruitDatas.csv')

# Лабораторная работа 1
# 1. Изучить распределение целевых классов и нескольких категориальных признаков.
sns.countplot(x='Class', data=data) #визуализация распределения целевого класса
sns.countplot(x='AREA', data=data) #визуализация категорийных признаков

# 2. Нарисовать распределения нескольких числовых признаков.
sns.histplot(data=data, x='PERIMETER') #визуализация распределения числовых признаков

# 3. Произвести нормализацию нескольких числовых признаков.
scaler = StandardScaler()
data[['PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS']] = scaler.fit_transform(data[['PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS']])

# 4. Посмотреть и визуализировать корреляцию признаков.
corr = data.corr(numeric_only=True)
#print(corr)
sns.heatmap(corr, cmap='coolwarm', annot=True)
#plt.show()