import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Загрузка данных из файла
data = pd.read_csv('FruitDatas.csv')

# Определение признаков (все колонки кроме Class)
features = data.columns[:-1]

# Разделение данных на обучающую и тестовую выборки в соотношении 70:30
X_train, X_test, y_train, y_test = train_test_split(data[features], data['Class'], test_size=0.3, random_state=42)

# Создание модели ансамбля случайных деревьев
model = RandomForestClassifier()

# Обучение модели на обучающей выборке
model.fit(X_train, y_train)

# Применение модели на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление метрик accuracy и precision
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

# Вывод метрик на экран
print("Accuracy:", accuracy)
print("Precision:", precision)

# Вычисление корреляционной матрицы
confusion = confusion_matrix(y_test, y_pred)

# Вывод корреляционной матрицы на экран
print("Confusion matrix:")
print(confusion)

# Подбор оптимальных параметров модели с помощью GridSearchCV
parameters = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
clf = GridSearchCV(RandomForestClassifier(), parameters)
clf.fit(X_train, y_train)

# Вывод лучших параметров на экран
print("Best parameters:")
print(clf.best_params_)

# Обучение модели на обучающей выборке с лучшими параметрами
best_model = RandomForestClassifier(n_estimators=clf.best_params_['n_estimators'], max_depth=clf.best_params_['max_depth'])
best_model.fit(X_train, y_train)

# Применение модели на тестовой выборке с лучшими параметрами
y_pred_best = best_model.predict(X_test)

# Вычисление метрик accuracy и precision с лучшими параметрами
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best, average='weighted')

# Вывод метрик с лучшими параметрами на экран
print("Accuracy with best parameters:", accuracy_best)
print("Precision with best parameters:", precision_best)