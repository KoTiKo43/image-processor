"""Загрузка и обработка изображения из датабазы."""
import csv
import json
import os
import time
from enum import Enum
from functools import wraps
from random import choice
from typing import Any, Callable, Dict, Optional

from artwork import Artwork

import cv2

import numpy as np

import requests


API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1/objects"


def timer(func: Callable) -> Callable:
    """
    Декоратор. Засекает время выполнения функции.

    Args:
        func (Callable): Функция для оборачивания.

    Returns:
        wrapper (Callable): Обёрнутая функция (с таймером).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):                           # noqa: ANN002, ANN003, ANN201
        """
        Оборачивает функцию таймером.

        Args:
            args: Аргументы оборачиваемой функции (распаковываются).
            kwargs: Аргументы оборачиваемой функции (распаковываются).

        Returns:
            result: Результат выполнения функции без обёртки.
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'Function {func.__name__} executed in {end - start:.4f} seconds')
        return result                                       # noqa: DCO030

    return wrapper                                          # noqa: DCO030


class OperationType(Enum):
    """Перечисление доступных операций обработки изображений."""
    GREYSCALE = "greyscale"
    BLUR = "blur"
    SOBEL = "sobel"
    GAMMA = "gamma"
    HISTOGRAM = "histogram"


class ImageProcessor:
    """
    Позволяет управлять обработкой изображений.
    """

    # Реестр операций: ключ - OperationType, значение - функция обработки
    _operations: Dict[OperationType, Callable] = {}

    def __init__(self, save_dir: str = "paintings") -> None:
        """
        Задание директории сохранения изображения.
        Создаёт директорию, если её ещё нет.
        Определяет стандартные операции.

        Args:
            save_dir (str): Название папки сохранения.
        """
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self._register_default_operations()

    @property
    def save_dir(self) -> str:
        """Доступ к директории сохранения (только для чтения)."""
        return self._save_dir

    def _register_default_operations(self) -> None:
        """Регистрирует стандартные операции."""
        self.register_operation(OperationType.GREYSCALE, self._greyscale_op)
        self.register_operation(OperationType.BLUR, self._blur_op)
        self.register_operation(OperationType.SOBEL, self._sobel_op)
        self.register_operation(OperationType.GAMMA, self._gamma_op)
        self.register_operation(OperationType.HISTOGRAM, self._histogram_op)

    def register_operation(self, op_type: OperationType, func: Callable) -> None:
        """
        Регистрирует новую операцию.

        Args:
            op_type (OperationType): Тип исполняемой операции.
            func (Callable): Исполняемая функция операции.
        """
        self._operations[op_type] = func

    @staticmethod
    def _greyscale_op(artwork: Artwork) -> np.ndarray:
        """Операция преобразования в градации серого."""
        return artwork.color_to_greyscale()

    @staticmethod
    def _blur_op(artwork: Artwork, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Операция сглаживания изображения."""
        return artwork.gaussian_blur(size=size, sigma=sigma)

    @staticmethod
    def _sobel_op(artwork: Artwork) -> np.ndarray:
        """Операция выделения границ с помощью оператора Собеля."""
        return artwork.sobel_transformation()

    @staticmethod
    def _gamma_op(artwork: Artwork, gamma: float = 1.0) -> np.ndarray:
        """Операция коррекции гаммы изображения."""
        return artwork.gamma_correction(gamma=gamma)

    @staticmethod
    def _histogram_op(artwork: Artwork) -> np.ndarray:
        """Операция выравнивания гистограммы изображения."""
        return artwork.histogram_alignment()

    @timer
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
            self.log("Dataset reading starts...")

            reader = csv.DictReader(f)
            paintings_ids = []
            for row in reader:
                if row['Classification'] == 'Paintings':
                    paintings_ids.append(int(row['Object ID']))

        self.log("Dataset has been readed")

        painting_id = choice(paintings_ids)
        painting_json = requests.get(API_BASE + f'/{painting_id}').json()
        while not (painting_json['isPublicDomain'] and painting_json['primaryImage']):
            painting_id = choice(paintings_ids)
            painting_json = requests.get(API_BASE + f'/{painting_id}').json()

        self.log("Random image chosen")

        painting_url = painting_json['primaryImageSmall' if small_image
                                     else 'primaryImage']

        painting_path = os.path.join(self.save_dir, f'{painting_id}.jpg')

        response = requests.get(painting_url)
        with open(painting_path, 'wb') as file:
            file.write(response.content)
            self.log("Image saved")

        json_path = os.path.join(self.save_dir, f"{painting_id}.json")
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(painting_json, jf, indent=2, ensure_ascii=False)
            self.log("JSON saved")

        image = cv2.imread(painting_path)
        with open(json_path, 'r') as f:
            metadata = json.load(f)
            self.log("Object created")

        return Artwork.create(image, metadata)

    @timer
    def process_artwork(self, artwork: Artwork,
                        operations: Dict[OperationType,
                                         Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Применяет список операций к изображению artwork.

        Args:
            artwork (Artwork): Объект искусства.
            operations (Dict): Словарь применяемых к изображению операций.
                            Ключ - название операции.
                            Значение - словарь переменных для функции.

        Raises:
            ValueError: Передаваемые операции должны существовать.

        Returns:
            results (Dict): Словарь с итоговыми изображениями.
        """
        results = {}

        self.log("Processing starts")

        for op_type, params in operations.items():
            if op_type not in self._operations:
                self.log("Error accured")
                raise ValueError(f"Operation {op_type} not exists!")
            # Вызов зарегистрированной функции с переданными параметрами
            self.log(f"Executing operation {op_type.value}...")
            results[op_type.value] = self._operations[op_type](artwork, **params)
            self.log(f"Operation {op_type.value} executed")

        self.log("Processing finished")

        return results

    def log(self, message: str) -> None:
        """
        Логирует события в консоли.

        Args:
            message (str): Сообщение для логирования.
        """
        print(f"[LOG][{time.strftime('%H:%M:%S')}] {message}")
