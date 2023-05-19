import pandas as pd

data = pd.read_csv('data2.csv')

candidate_stats = {
    'Количество откликов': len(data),
    'Количество релевантных откликов': len(data[data['Статус'] == 'прошел']),
    'Количество нерелевантных откликов': len(data[data['Статус'] == 'не прошел']),
    'Средний возраст': data['Возраст'].mean(),
    'Города': ', '.join(data['Город'].unique()),
    'Вузы': ', '.join(data['Вуз'].unique()),
    'Образование': ', '.join(data['Образование'].unique()),
    'Направления стажировки': ', '.join(data['Направление стажировки'].unique()),
    'Каналы привлечения': ', '.join(data['Канал привлечения'].unique())
}

internship_data = data[data['Категория'] == 'комплексы Правительства Москвы']
passed_internship_data = internship_data[internship_data['Статус'] == 'прошел']

internship_stats = {
    'Количество заявок на стажеров (комплексы Правительства Москвы)': len(internship_data),
    'Количество прошедших стажировку (комплексы Правительства Москвы)': len(passed_internship_data),
    'Количество заявок от органов власти и учреждений': len(data[data['Категория'] == 'органы власти и учреждения']),
}

education_stats = passed_internship_data['Образование'].value_counts().to_dict()
internship_stats['Статистика по образованию'] = education_stats

task_stats = passed_internship_data['Задачи стажеров'].value_counts().to_dict()
internship_stats['Статистика по задачам стажеров'] = task_stats

print('Статистика по заявкам кандидатов:')
for key, value in candidate_stats.items():
    print(f'{key}: {value}')

print('\nСтатистика по заявкам на стажеров:')
for key, value in internship_stats.items():
    print(f'{key}: {value}')
