import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.models import load_model

model = load_model('model.h5')

onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
scaler = StandardScaler()

categorical_columns = ['Вуз', 'Образование', 'Город', 'Возраст', 'Канал привлечения']
numerical_columns = ['Возраст']

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

new_categorical_encoded = onehot_encoder.fit_transform(new_data_encoded[categorical_columns])
new_categorical_df = pd.DataFrame(new_categorical_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns))
new_data_encoded = pd.concat([new_categorical_df, new_data_encoded[numerical_columns]], axis=1)

new_data_encoded[numerical_columns] = scaler.fit_transform(new_data_encoded[numerical_columns])

probability = model.predict(new_data_encoded)
prediction = probability[0][0]

print(f'Стажер {new_data.iloc[0]["Имя"]} может пройти стажировку с вероятностью {prediction * 100:.2f}%')