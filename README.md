# <a id="readme-top"></a>

<h1 align="center">HSE Sber RecSys Hack</h1>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contacts">Contacts</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

В рамках хакатона мы построили алгоритм персонализированных рекомендаций, обогащенных кросс-доменными данными, который позволяет, например, на основе действий пользователя в интернет-магазине предсказать, какую музыку ему предложить в стриминговом сервисе.

В проекте мы используем реальные данные экосистемы Сбера, включая интеракции пользователей на сервисах **МегаМаркет** и **Звук**. Наш алгоритм проводит отбор кандидатов с помощью модели ALS, обученной на объединенных пользовательских взаимодействиях, а затем ранжирует результаты с помощью LightGBM Ranker. 

Уникальность нашего решения заключается разделении польозователей на теплых и холодных: холодным пользователям,  для которых недостаточно данных, вместо персонализированных рекомендаций предлагаются топ-10 популярных товаров. 

Стек технологий: Python, NumPy, Pandas, Implicit ALS, LightGBM Ranker.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Обучение модели:

1. Подготовьте файлы с данными:
    - `train_smm.parquet` (взаимодействия пользователей с МегаМаркет)
    - `train_zvuk.parquet` (взаимодействия пользователей с Звук)

2. Запустите скрипт обучения:
    ```bash
    python train_predict.py
    ```
    Это создаст и сохранит обученную модель ALS в папке `saved_models`.

### Инференс (предсказание рекомендаций):

1. Убедитесь, что модель обучена и сохранена в `saved_models`.
2. Подготовьте тестовые данные (`test_smm.parquet` или `test_zvuk.parquet`).
3. Запустите предсказание:
    ```bash
    python train_predict.py
    ```
    Результаты рекомендаций сохранятся в файлы `submission_smm.parquet` и `submission_zvuk.parquet`.

### Основные файлы:
- **train_predict.py** - скрипт для обучения модели и генерации рекомендаций.
- **my_model.py** - реализация модели ALS и вспомогательные методы.
- **saved_models/** - директория для хранения обученной модели.
- **data/** - директория с исходными и тестовыми данными.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contacts

Tatiana Irincheeva - [@iamtanyaaaaa](https://t.me/@iamtanyaaaaa) - tairincheeva@gmail.com


Project Link: [https://github.com/iamtanyaaa/Hack-RecSys](https://github.com/iamtanyaaa/Hack-RecSys)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

