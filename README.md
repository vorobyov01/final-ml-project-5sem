# Практическое задание 3

Данный репозиторий содержит проект сайта, через который можно обучать случайный лес и градиентный бустинг для решения задачи регрессии.

### 1. Сборка

Для сборки контейнера из корневой директории:

```bash
sh scripts/build.sh
```

Для запуска:

```
sh scripts/run.sh
```

### 2. Описание работы

Сайт состоит из 3-х основных разделов: создание модели и выбор параметров, обучение модели, предсказание на новых данных. Рассмотрим их по порядку.

**a. Создание модели и выбор параметров**
![picture](https://drive.google.com/uc?export=view&id=15GqB2msBNjioq3nBTKM5wvB4mkwHNJ0q)

В начале предлагается выбрать данные для тренировки. Подходят только csv файлы с колонкой `price` в качестве целевой и с колонкой `date`. 
Далее предлагается выбрать параметры для нашей модели и саму модель – `Random Forest` или `Gradient Boosting`. Выбор параметров совпадает с экспериментами – `n_estimators`, `feature_subsample_size`, `max_depth`, `learning_rate`. Они изначально заданы по-умолчанию, однако их можно переписать. В случае с `Random Forest` параметр `learning_rate` будет проигнорирован.

**b. Тренировка модели**

![picture](https://drive.google.com/uc?export=view&id=1ryo4N0dWJ0prFfkvMMZdCSrCr36qVAgA)

На данной странице есть кнопка "Train", после нажатия на которую начнется процесс обучения с ранее выбранными параметрами на заданных данных. После обучения ниже появится кнопка "Go to predicting page" при нажатии на которую можно перейти на страницу с предсказанием данных на уже обученной модели.

**c. Предсказание на новых данных**

![picture](https://drive.google.com/uc?export=view&id=1T6FN0RLpatKzvL37EfGeMkw9eXaCdhEc)

На этой странице пользователю снова предлагается загрузить датасет такого же формата (можно с целевой переменной или без). Далее можно получить ссылку на скачивание файла.

Стоит отметить, что данный сайт предполагает только однопользовательский режим. В рабочем каталоге будет создаваться файл `submission.csv`, который не будет автоматически удаляться, лишь перезаписываться.
