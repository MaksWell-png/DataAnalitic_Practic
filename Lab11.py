import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt

# Загрузка данных из файла
data = pd.read_csv('FruitDatas.csv')

# Определение признаков (все колонки кроме Class)
features = data.columns[:-1]

# Разделение данных на обучающую и тестовую выборки в соотношении 70:30
X_train, X_test, y_train, y_test = train_test_split(data[features], data['Class'], test_size=0.3, random_state=42)

# Преобразование меток классов в числовые значения
class_mapping = {label: idx for idx, label in enumerate(set(data['Class']))}
y_train = y_train.map(class_mapping)
y_test = y_test.map(class_mapping)

# Создание модели нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(class_mapping), activation='softmax')
])

# Компиляция модели с оптимизатором Adam и функцией потерь sparse_categorical_crossentropy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели на обучающей выборке с сохранением истории изменения значения функции потерь
history = model.fit(X_train, y_train,
                    epochs=50, batch_size=32,
                    validation_data=(X_test, y_test))

# Вывод графика изменения значения функции потерь
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Применение модели на тестовой выборке
y_pred = model.predict_classes(X_test)

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

# Подбор оптимальных параметров модели не требуется, так как мы использовали модель с параметрами по умолчанию.