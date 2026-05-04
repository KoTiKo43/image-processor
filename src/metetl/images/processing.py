"""Загрузка и обработка изображения из датабазы."""
import asyncio
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from random import choice
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

import aiofiles

import aiohttp

import cv2

from models import Artwork

import numpy as np

import requests

from ..decorators import async_timer, timer
from ..logging_config import logger


API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
DEFAULT_SAVE_DIR = "images"


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

    def __init__(self, save_dir: str = DEFAULT_SAVE_DIR) -> None:
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


class AsyncImagePipeline:
    """Асинхронный пайплайн: скачивание -> обработка -> сохранение."""
    def __init__(self, save_dir: str = DEFAULT_SAVE_DIR, engine: str = "python") -> None:
        """
        Инициализация.

        Args:
            save_dir (str): Путь к папке для сохранения изображений.
            engine (str): Способ обработки (чистый пайтон или с помощью Си).
        """
        self.save_dir = save_dir
        self.engine = engine  # "python" или "c"
        os.makedirs(save_dir, exist_ok=True)
        # Если выбран С-движок, убедимся что расширение доступно
        if engine == "c":
            try:
                from metetl.images.c_ext import convolve2d as _  # noqa: F401
            except ImportError:
                logger.warning("C extension not available, falling back to python engine")
                self.engine = "python"

    async def download_one(
        self, session: aiohttp.ClientSession, obj_id: int, index: int
    ) -> Optional[Tuple[int, int, str, str]]:
        """
        Асинхронно скачивает одно изображение.

        Args:
            session: Сессия клиента.
            obj_id: ID изображения.
            index: Индекс изображения.

        Returns:
            Кортеж (index, obj_id, img_path, json_path) или None.
        """
        try:
            async with session.get(f"{API_BASE}/{obj_id}") as resp:
                painting_json = await resp.json()

            if not (painting_json.get("isPublicDomain") and painting_json.get("primaryImage")):  # noqa E501
                logger.debug(f"Image {obj_id} not public domain, skipping")
                return None

            original_dir = os.path.join(self.save_dir, "original")
            os.makedirs(original_dir, exist_ok=True)

            img_path = os.path.join(original_dir, f"{index}_{obj_id}_original.jpg")
            json_path = os.path.join(original_dir, f"{index}_{obj_id}_metadata.json")

            async with aiofiles.open(json_path, "w", encoding="utf-8") as jf:
                await jf.write(json.dumps(painting_json, indent=2, ensure_ascii=False))

            img_url = painting_json["primaryImage"]
            async with session.get(img_url) as img_resp:
                async with aiofiles.open(img_path, "wb") as f:
                    await f.write(await img_resp.read())

            logger.info(f"Downloaded image {index} (ID: {obj_id})")
            return index, obj_id, img_path, json_path

        except Exception as e:
            logger.error(f"Download failed for object {obj_id}: {e}")
            return None

    async def download_generator(
        self, ids: List[int]
    ) -> AsyncGenerator[Tuple[int, int, str, str], None]:
        """
        Асинхронный генератор скачивания.
        Отдаёт результаты по мере завершения каждого запроса.

        Args:
            ids: Список айди.

        Yields:
            result: Словарь с результатом скачивания.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.download_one(session, obj_id, idx)
                for idx, obj_id in enumerate(ids, start=1)
            ]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is not None:
                    yield result

    def process_single(self, args: Tuple[int, int, str, str]
                       ) -> Optional[Tuple[int, int, dict]]:
        """
        Обработка одного изображения в отдельном процессе.
        Возвращает (index, obj_id, dict с изображениями).

        Args:
            args: Индекс, айди, пути к изображению и метаданным.

        Returns:
            Индекс, айди, словарь обработанных изображений.
        """
        index, obj_id, img_path, json_path = args
        logger.debug(f"Processing image {index} (ID={obj_id}) in PID {os.getpid()}")
        try:
            image = cv2.imread(img_path)
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            artwork = Artwork.create(image, metadata)

            # Стандартный набор операций
            results = {
                "greyscale": artwork.color_to_greyscale(),
                "blur": artwork.gaussian_blur(size=5, sigma=1.0),
                "sobel": artwork.sobel_transformation(),
                "gamma": artwork.gamma_correction(gamma=1.5),
                "histogram": artwork.histogram_alignment(),
            }

            logger.info(f"Processed image {index} (ID={obj_id})")

            return index, obj_id, results

        except Exception as e:
            logger.error(f"Processing failed for image {index} (ID={obj_id}): {e}")
            return None

    async def processing_generator(
        self, source: AsyncGenerator[Tuple[int, int, str, str], None]
    ) -> AsyncGenerator[Tuple[int, int, dict], None]:
        """
        Генератор обработки.

        Args:
            source: Асинхронный генератор с данными об изображении.

        Yields:
            item: Кортеж с индексом, ID объекта и словарём обработанных изображений.
        """
        loop = asyncio.get_running_loop()
        result_queue = asyncio.Queue()
        executor = ProcessPoolExecutor(max_workers=os.cpu_count())

        async def enqueue_future(future: asyncio.Future) -> None:
            """
            Ждёт завершения обработки и отправляет результат в очередь.

            Args:
                future: Объект Future.
            """
            res = await future
            if res is not None:
                await result_queue.put(res)

        async def process_in_pool() -> None:
            """
            Запускает мультипроцессорную обработку изображений.
            """
            async with asyncio.TaskGroup() as tg:
                async for item in source:
                    future = loop.run_in_executor(executor, self.process_single, item)
                    tg.create_task(enqueue_future(future))

            await result_queue.put(None)  # сигнал завершения

        pool_task = asyncio.create_task(process_in_pool())

        while True:
            item = await result_queue.get()
            if item is None:
                break

            yield item

        await pool_task
        executor.shutdown(wait=True)

    async def save_generator(
        self, source: AsyncGenerator[Tuple[int, int, dict], None]
    ) -> AsyncGenerator[str, None]:
        """
        Генератор сохранения.
        Сохраняет файлы неблокирующе через asyncio.to_thread.

        Args:
            source: Асинхронный генератор с данными об изображении.

        Yields:
            str: Строку о том, что файл сохранён.
        """
        proc_dir = os.path.join(self.save_dir, "processed")
        os.makedirs(proc_dir, exist_ok=True)

        async for index, obj_id, results in source:
            for mode, img in results.items():
                fname = f"{index}_{obj_id}_{mode}.jpg"
                path = os.path.join(proc_dir, fname)
                await asyncio.to_thread(cv2.imwrite, path, img)

            logger.info(f"Saved processed image {index} (ID={obj_id})")

            yield f"Image {index} (ID={obj_id}) saved"

    @async_timer
    async def run_pipeline(self, ids: List[int]) -> None:
        """
        Главная корутина: запускает весь пайплайн.

        Args:
            ids: Айди изображений для обработки.
        """
        logger.info(f"Starting async pipeline for {len(ids)} images (engine={self.engine})")  # noqa E501
        download_gen = self.download_generator(ids)
        processing_gen = self.processing_generator(download_gen)
        async for msg in self.save_generator(processing_gen):
            logger.debug(f"Pipeline progress: {msg}")

        logger.info("Async pipeline completed")
