"""
Скачивает и добавляет в папку изображение случайной картины из датасета
"""

import csv
import os
import random

import requests

DATASET = "MetObjects.csv"
SAVE_DIR = "paintings"
API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1/objects"


def get_paintings_ids(dataset_path: str) -> list:
    """
    Возвращает список, содержащий id всех картин.

    Args:
        dataset_path (str): Путь к текстовому файлу датасета.

    Returns:
        list: Список integer чисел - id картин.
    """

    with open(dataset_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        paintings_ids = []
        for row in reader:
            if row['Classification'] == 'Paintings':
                paintings_ids.append(int(row['Object ID']))

    return paintings_ids


def get_painting_json(painting_id: int) -> dict:
    """
    Извлекает json для определённой картины

    Args:
        painting_id (int): id картины.

    Returns:
        dict: JSON-ответ, содержащий информацию о картине.
    """

    response = requests.get(API_BASE + f'/{painting_id}')
    return response.json()


def download_image(url: str, image_path: str) -> None:
    """
    Скачивает и сохраняет изображение по ссылке

    Args:
        url (str): Ссылка на скачивание.
        image_path (str): Путь для сохранения изображения.
    """

    response = requests.get(url)
    with open(image_path, 'wb') as file:
        file.write(response.content)


# Создаём папку для картин
os.makedirs(SAVE_DIR, exist_ok=True)

paintings_ids = get_paintings_ids(DATASET)
painting_id = random.choice(paintings_ids)

painting_json = get_painting_json(painting_id)
while not painting_json['isPublicDomain']:
    painting_id = random.choice(paintings_ids)
    painting_json = get_painting_json(painting_id)

painting_url = painting_json['primaryImage']

painting_path = os.path.join(SAVE_DIR, f'{painting_id}.jpg')
download_image(painting_url, painting_path)
