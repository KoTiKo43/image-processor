"""Загрузка и обработка изображения из датабазы."""
import asyncio
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from random import choice
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiofiles

import aiohttp

import cv2

from models import Artwork

import numpy as np

import requests

from ..decorators import timer
from ..logging_config import logger


API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1/objects"


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

    def __init__(self, save_dir: str = "images") -> None:
        """
        Задание директории сохранения изображения.
        Создаёт директорию, если её ещё нет.
        Определяет стандартные операции.

        Args:
            save_dir (str): Название папки сохранения.
        """
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self._operations: Dict[OperationType, callable] = {}
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
        try:
            paintings_ids = []

            logger.debug("Reading dataset for random image download")

            with open(dataset_path, 'r', encoding='utf-8-sig') as f:
                logger.info("Dataset reading starts...")

                reader = csv.DictReader(f)
                paintings_ids = []
                for row in reader:
                    if row['Classification'] == 'Paintings':
                        paintings_ids.append(int(row['Object ID']))

            logger.debug(f"Found {len(paintings_ids)} paintings in dataset")

            painting_id = choice(paintings_ids)
            painting_json = requests.get(API_BASE + f'/{painting_id}').json()
            while not (painting_json['isPublicDomain'] and painting_json['primaryImage']):
                painting_id = choice(paintings_ids)
                painting_json = requests.get(API_BASE + f'/{painting_id}').json()

            logger.info("Random image chosen")

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

            logger.info(f"Downloaded image {painting_id}")
            return Artwork.create(image, metadata)

        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            return None

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

        logger.info("Image processing starts")

        for op_type, params in operations.items():
            if op_type not in self._operations:
                logger.error(f"Operation {op_type} not registered")
                raise ValueError(f"Operation {op_type} not exists!")

            logger.debug(f"Executing operation {op_type.value}...")
            try:
                results[op_type.value] = self._operations[op_type](artwork, **params)
            except Exception as e:
                logger.error(f"Error in {op_type.value}: {e}")
                raise

            logger.info(f"Operation {op_type.value} executed")

        logger.info("Processing finished")

        return results


class AsyncImageProcessor:
    """Класс для асинхронного скачивания и обработки"""
    def __init__(self, save_dir: str = "images", max_workers: int = 1) -> None:
        """
        Инициализация save_dir

        Args:
            save_dir (str): Путь к папке для сохранения изображений.
            max_workers (int): Максимальное количество процессов.
        """
        self.save_dir = save_dir
        self.max_workers = max_workers or os.cpu_count()
        os.makedirs(save_dir, exist_ok=True)

    async def download_one(self, session: aiohttp.ClientSession,
                           object_id: int, index: int) -> tuple:
        """
        Скачивает изображение и его метаданные.

        Args:
            session: Асинхронная клиентская сессия.
            object_id: ID изображения.
            index: Порядковый номер изображения.

        Returns:
            None: Если изображение не в публичном доступе.
            index, object_id, painting_path, json_path: Если всё прошло успешно.
        """
        try:
            logger.debug(f"Downloading image {index} (ID: {object_id})")
            async with session.get(f"{API_BASE}/{object_id}") as resp:
                painting_json = await resp.json()

            if not (painting_json.get('isPublicDomain') and painting_json.get('primaryImage')):  # noqa E501
                logger.debug(f"Image {object_id} not public domain, skipping")
                return None

            original_dir = os.path.join(self.save_dir, "original")
            os.makedirs(original_dir, exist_ok=True)
            img_path = os.path.join(original_dir, f"{index}_{object_id}_original.jpg")
            json_path = os.path.join(original_dir, f"{index}_{object_id}_metadata.json")

            async with aiofiles.open(json_path, 'w', encoding='utf-8') as jf:
                await jf.write(json.dumps(painting_json, indent=2, ensure_ascii=False))

            async with session.get(painting_json['primaryImage']) as img_resp:
                async with aiofiles.open(img_path, 'wb') as f:
                    await f.write(await img_resp.read())

            logger.info(f"Downloaded image {index} (ID: {object_id})")
            return index, object_id, img_path, json_path
        except Exception as e:
            logger.error(f"Download failed for {object_id}: {e}")
            return None

    async def download_batch(self, ids: List[int]) -> List[tuple]:
        """
        Пакетная асинхронная загрузка.

        Args:
            ids: Список айди изображений.

        Returns:
            List: Список результатов выполненных задач.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self.download_one(session, object_id, i + 1)
                     for i, object_id in enumerate(ids)]
            results = await asyncio.gather(*tasks)

        return [r for r in results if r is not None]

    def process_single(self, args: Tuple[int, int, str, str]) -> int:
        """
        Функция обработки в отдельном процессе.

        Args:
            args: Список с индексом, айди, путём к изображению и метаданным.

        Returns:
            index: Индекс изображения.
        """
        index, obj_id, img_path, json_path = args
        logger.debug(f"Processing image {index} (ID: {obj_id}) in PID {os.getpid()}")
        try:
            image = cv2.imread(img_path)
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            artwork = Artwork.create(image, metadata)

            # Применяем стандартные операции
            proc_dir = os.path.join(self.save_dir, "processed")
            os.makedirs(proc_dir, exist_ok=True)
            ops = {
                'greyscale': artwork.color_to_greyscale(),
                'blur': artwork.gaussian_blur(size=5, sigma=1.0),
                'sobel': artwork.sobel_transformation(),
                'gamma': artwork.gamma_correction(gamma=1.5),
                'histogram': artwork.histogram_alignment()
            }

            for name, img_array in ops.items():
                fname = f"{index}_{obj_id}_{name}.jpg"
                cv2.imwrite(os.path.join(proc_dir, fname), img_array)

            logger.info(f"Processed image {index} (ID: {obj_id})")
            return index

        except Exception as e:
            logger.error(f"Processing failed for {index}: {e}")
            return None

    def run_processing(self, download_results: List[tuple]) -> None:
        """
        Запуск параллельной обработки скачанных изображений.

        Args:
            download_results: Результаты скачанных изображений
        """
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_single, item)
                       for item in download_results]
            for future in futures:
                future.result()

    @timer
    async def full_pipeline(self, ids: List[int]) -> None:
        """
        Полный асинхронный пайплайн.

        Args:
            ids: Список с айди изображений.
        """
        logger.info(f"Starting async pipeline for {len(ids)} images")
        downloads = await self.download_batch(ids)
        if not downloads:
            logger.warning("No images downloaded")
            return

        loop = asyncio.get_running_loop()
        # Запускаем обработку в отдельных процессах
        await loop.run_in_executor(None, self.run_processing, downloads)
        logger.info("Async pipeline completed")
