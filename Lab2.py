import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Загрузка датасета из CSV-файла
df = pd.read_csv('FruitDatas.csv')

# Разделение датасета на обучающую и тестовую выборки
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели наивного байесовского классификатора с параметрами по умолчанию
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Предсказание классов для тестовой выборки
y_pred = nb_model.predict(X_test)

# Вычисление метрик качества модели
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
confusion_mat = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Confusion matrix:\n', confusion_mat)

# Подбор оптимальных параметров модели с помощью GridSearchCV
parameters = {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]}
nb_model_gs = GridSearchCV(GaussianNB(), parameters)
nb_model_gs.fit(X_train, y_train)

print('Best parameters:', nb_model_gs.best_params_)
print('Best score:', nb_model_gs.best_score_)