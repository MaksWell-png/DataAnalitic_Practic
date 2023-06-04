import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Загрузка датасета из CSV-файла
df = pd.read_csv('FruitDatas.csv')

# Разделение датасета на обучающую и тестовую выборки
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели SVM с параметрами по умолчанию
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Предсказание классов для тестовой выборки
y_pred = svm_model.predict(X_test)

# Вычисление метрик качества модели
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
confusion_mat = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Confusion matrix:\n', confusion_mat)

# Подбор оптимальных параметров модели с помощью GridSearchCV
parameters = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
svm_model_gs = GridSearchCV(SVC(), parameters)
svm_model_gs.fit(X_train, y_train)

print('Best parameters:', svm_model_gs.best_params_)
print('Best score:', svm_model_gs.best_score_)