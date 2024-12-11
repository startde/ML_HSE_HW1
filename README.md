# Выводы по проделанной работе
## Обзор проекта

Проект был направлен на создание модели машинного обучения для предсказания стоимости автомобилей и разработку веб-сервиса на базе FastAPI для предоставления этих предсказаний.

## Что было сделано:
- Подготовлены и проанализированы данные.
- Обучены различные модели машинного обучения.
- Проведена настройка гиперпараметров с использованием GridSearchCV.
- Выбрана лучшая модель на основе метрик качества.
- Разработана архитектура REST API на FastAPI для предоставления сервиса предсказания.

## Достигнутые результаты:
- Модель: разработана модель, способная предсказывать стоимость автомобиля с высокой точностью. Лучшие результаты показала регрессия с регуляризацией Ridge.

- API: Создан гибкий REST API, который поддерживает два типа запросов:

- Индивидуальное предсказание: принимает JSON с характеристиками автомобиля и возвращает предсказанную стоимость.

- CSV-предсказание: принимает CSV файл с характеристиками множества автомобилей и возвращает CSV файл с предсказанными стоимостями.

## Метрики качества модели:

- R² на тестовой выборке: 0.8521
- MSE на тестовой выборке: 0.0952 (после логарифмирования)
- Эти показатели свидетельствуют о высокой точности модели.

## Что дало наибольший буст в качестве:
- Использование регрессии с регуляризацией Ridge.
- Использование GridSearchCV для настройки параметров.
- Обработка категориальных признаков и таргет параметра.

## Ограничения и проблемы:
- Сложность задачи: предсказание стоимости автомобилей оказалось сложным из-за множества факторов, влияющих на цену.
- Качество данных: наличие шума в данных и недостаток некоторых характеристик ограничивает точность модели.

## Что сделать не вышло и почему:
- Не удалось достичь идеальной точности предсказания.
- Не удалось оценить переобучение

# Заключение
- Проект успешно достиг поставленных целей: разработан сервис на базе FastAPI, обеспечивающий точные предсказания стоимости автомобилей.

# Скрины работы сервиса
/FastApi_predict_item.png
/FastApi_predict_items.png