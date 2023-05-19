import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv('data.csv')

label_encoder = LabelEncoder()

#data['Статус'] = label_encoder.fit_transform(data['Статус'])

data['Статус'] = (data['Статус'] == 'прошел')
categorical_columns = ['Вуз', 'Образование', 'Город', 'Возраст', 'Канал привлечения']
numerical_columns = ['Возраст']

onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_encoded = onehot_encoder.fit_transform(data[categorical_columns])
categorical_df = pd.DataFrame(categorical_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns))
data_encoded = pd.concat([categorical_df, data[numerical_columns]], axis=1)

scaler = StandardScaler()
data_encoded[numerical_columns] = scaler.fit_transform(data[numerical_columns])

X = data_encoded
y = data['Статус']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Точность модели: %.2f%%' % (accuracy * 100))

new_data = pd.DataFrame({
    'Фамилия': ['Сабиров'],
    'Имя': ['Айдар'],
    'Отчество': ['Рустамович'],
    'Возраст': [25.0],
    'Город': ['Санкт-Петербург'],
    'Вуз': ['СПбПУ'],
    'Образование': ['Высшее'],
    'Направление стажировки': ['IT'],
    'Канал привлечения': ['Рекомендация от учителя']
})
new_data_encoded = new_data.copy()

new_categorical_encoded = onehot_encoder.transform(new_data_encoded[categorical_columns])
new_categorical_df = pd.DataFrame(new_categorical_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns))
new_data_encoded = pd.concat([new_categorical_df, new_data_encoded[numerical_columns]], axis=1)

new_data_encoded[numerical_columns] = scaler.transform(new_data_encoded[numerical_columns])

probability = model.predict(new_data_encoded)
prediction = probability[0][0]

print(f'Стажер {new_data.iloc[0]["Имя"]} может пройти стажировку с вероятностью {prediction * 100:.2f}%')