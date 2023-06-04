import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Загрузка датасета из CSV-файла
df = pd.read_csv('FruitDatas.csv')

# Разделение датасета на обучающую и тестовую выборки
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели KNN с параметрами по умолчанию
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Предсказание классов для тестовой выборки
y_pred = knn_model.predict(X_test)

# Вычисление метрик качества модели
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
confusion_mat = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Confusion matrix:\n', confusion_mat)

# Подбор оптимальных параметров модели с помощью GridSearchCV
parameters = {'n_neighbors': [3, 5, 7, 9]}
knn_model_gs = GridSearchCV(KNeighborsClassifier(), parameters)
knn_model_gs.fit(X_train, y_train)

print('Best parameters:', knn_model_gs.best_params_)
print('Best score:', knn_model_gs.best_score_)