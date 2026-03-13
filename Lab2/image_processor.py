"""Загрузка и обработка изображения из датабазы."""
import csv
import json
import os
from functools import wraps
from random import choice
from typing import Any, Dict, Optional

from artwork import Artwork

import cv2

import numpy as np

import requests


API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1/objects"


class ImageProcessor:
    """
    Позволяет скачать и обработать изображение.
    """
    def __init__(self, save_dir: str = "paintings") -> None:
        """
        Задание директории сохранения изображения.
        Создаёт директорию, если её ещё нет.

        Args:
            save_dir (str): Название папки сохранения.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def download_random_image(self, dataset_path: str = "MetObjects.csv",
                              small_image: bool = False) -> Optional[Artwork]:
        """
        Загружает в директорию изображение случайной картины из датасета.

        Args:
            dataset_path (str): Путь к текстовому файлу датасета.
            small_image (bool): Флаг для скачивания сжатой версии изображения.

        Returns:
            Artwork: Объект типа Artwork, содержащий изображение и его метаданные.
        """
        paintings_ids = []

        with open(dataset_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            paintings_ids = []
            for row in reader:
                if row['Classification'] == 'Paintings':
                    paintings_ids.append(int(row['Object ID']))

        painting_id = choice(paintings_ids)
        painting_json = requests.get(API_BASE + f'/{painting_id}').json()
        while not (painting_json['isPublicDomain'] and painting_json['primaryImage']):
            painting_id = choice(paintings_ids)
            painting_json = requests.get(API_BASE + f'/{painting_id}').json()

        painting_url = painting_json['primaryImageSmall' if small_image
                                     else 'primaryImage']

        painting_path = os.path.join(self.save_dir, f'{painting_id}.jpg')

        response = requests.get(painting_url)
        with open(painting_path, 'wb') as file:
            file.write(response.content)

        json_path = os.path.join(self.save_dir, f"{painting_id}.json")
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(painting_json, jf, indent=2, ensure_ascii=False)

        image = cv2.imread(painting_path)
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        return Artwork(image, metadata)

    def process_artwork(self, artwork: Artwork,
                        operations: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Применяет список операций к изображению.

        Args:
            artwork (Artwork): Объект искусства.
            operations (Dict): Словарь применяемых к изображению операций.
                            Ключ - название операции.
                            Значение - словарь переменных для функции.

        Returns:
            results (Dict): Словарь с итоговыми изображениями.
        """
        results = {}

        if 'greyscale' in operations or 'grayscale' in operations:
            results['greyscale'] = artwork.color_to_greyscale()

        if 'blur' in operations:
            params = operations['blur']
            # Используем параметры, если они есть, иначе значения по умолчанию
            size = params.get('size', 5)
            sigma = params.get('sigma', 1.0)
            results['blur'] = artwork.gaussian_blur(size, sigma)

        if 'sobel' in operations:
            results['sobel'] = artwork.sobel_transformation()

        if 'gamma' in operations:
            params = operations['gamma']
            gamma = params.get('gamma', 1.0)
            results['gamma'] = artwork.gamma_correction(gamma)

        if 'hist' in operations:
            results['hist'] = artwork.histogram_alignment()

        return results
