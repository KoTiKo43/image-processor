"""Загрузка и обработка изображения из датабазы."""
import asyncio
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from random import choice
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional, Tuple

import aiofiles

import aiohttp

import cv2

from metetl.decorators import async_timer, timer
from metetl.images.models import Artwork
from metetl.logging_config import logger

import numpy as np

import requests


API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
DEFAULT_SAVE_DIR = "images"


class OperationType(Enum):
    """Перечисление доступных операций обработки изображений."""
    GREYSCALE = "greyscale"
    BLUR = "blur"
    SOBEL = "sobel"
    GAMMA = "gamma"
    HISTOGRAM = "histogram"


default_operations: Dict[OperationType, Dict[str, Any]] = {
    OperationType.GREYSCALE: {},
    OperationType.BLUR: {"size": 5, "sigma": 1.0},
    OperationType.SOBEL: {},
    OperationType.GAMMA: {"gamma": 1.5},
    OperationType.HISTOGRAM: {}
}


class ImagePipeline:
    """
    Позволяет управлять обработкой изображений.
    """
    def __init__(self, save_dir: str = DEFAULT_SAVE_DIR, engine: str = "python",
                 operations: Optional[Dict[OperationType, Dict[str, Any]]] = None,
                 ) -> None:
        """
        Задание директории сохранения изображения.
        Создаёт директорию, если её ещё нет.
        Определяет стандартные операции.

        Args:
            save_dir (str): Название папки сохранения.
            engine (str): Способ обработки (python или c).
            operations: Операции, которые нужно проделать с изображением.
        """
        self.save_dir = save_dir
        self.engine = engine
        self.operations = default_operations if not operations else operations
        os.makedirs(save_dir, exist_ok=True)

    def download_generator(self, ids: List[int]
                           ) -> Generator[Tuple[int, int, str, str], None, None]:
        """
        Генератор синхронной загрузки.

        Args:
            ids: Список айди изображений.

        Yields:
            Индекс, айди объекта, пути к изображению и метаданным.
        """
        for idx, obj_id in enumerate(ids, start=1):
            try:
                resp = requests.get(f"{API_BASE}/{obj_id}")
                painting_json = resp.json()
                if not (painting_json.get('isPublicDomain') and painting_json.get('primaryImage')):  # noqa E501
                    logger.debug(f"Image {obj_id} not public domain, skipping")
                    continue

                orig_dir = os.path.join(self.save_dir, "original")
                os.makedirs(orig_dir, exist_ok=True)
                img_path = os.path.join(orig_dir, f"{idx}_{obj_id}_original.jpg")
                json_path = os.path.join(orig_dir, f"{idx}_{obj_id}_metadata.json")

                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump(painting_json, jf, indent=2, ensure_ascii=False)

                img_resp = requests.get(painting_json['primaryImage'])
                with open(img_path, 'wb') as f:
                    f.write(img_resp.content)

                logger.info(f"Downloaded image {idx} (ID: {obj_id})")
                yield idx, obj_id, img_path, json_path

            except Exception as e:
                logger.error(f"Download failed for {obj_id}: {e}")

    def processing_generator(self, source: Generator) -> Generator[Tuple[int, int, Dict[str, np.ndarray]], None, None]:  # noqa E501
        """
        Генератор обработки (последовательная, без процессов).

        args:
            source: Генератор с индексом, айди объекта, путью к изображению и метаданным.

        Yields:
            Индекс, айди объекта, результат обработки.
        """
        for index, obj_id, img_path, json_path in source:
            logger.debug(f"Processing image {index} (ID={obj_id})")
            try:
                image = cv2.imread(img_path)
                with open(json_path, 'r') as f:
                    metadata = json.load(f)

                artwork = Artwork.create(image, metadata)

                results = {}
                for op_type, params in self.operations.items():
                    if op_type == OperationType.GREYSCALE:
                        results['greyscale'] = artwork.color_to_greyscale()
                    elif op_type == OperationType.BLUR:
                        results['blur'] = artwork.gaussian_blur(**params)
                    elif op_type == OperationType.SOBEL:
                        results['sobel'] = artwork.sobel_transformation()
                    elif op_type == OperationType.GAMMA:
                        results['gamma'] = artwork.gamma_correction(**params)
                    elif op_type == OperationType.HISTOGRAM:
                        results['histogram'] = artwork.histogram_alignment()

                logger.info(f"Processed image {index} (ID={obj_id})")

                yield index, obj_id, results
            except Exception as e:
                logger.error(f"Processing failed for {index}: {e}")

    def save_generator(self, source: Generator) -> Generator[str, None, None]:
        """
        Генератор сохранения.

        Args:
            source: Индекс, айди, результат.

        Yields:
            Строку с сообщением о сохранении изображения.
        """
        proc_dir = os.path.join(self.save_dir, "processed")
        os.makedirs(proc_dir, exist_ok=True)
        for index, obj_id, results in source:
            for mode, img in results.items():
                fname = f"{index}_{obj_id}_{mode}.jpg"
                cv2.imwrite(os.path.join(proc_dir, fname), img)

            logger.info(f"Saved processed image {index} (ID={obj_id})")
            yield f"Image {index} (ID={obj_id}) saved"

    @timer
    def run_pipeline(self, ids: List[int]) -> None:
        """
        Запуск пайплайна.

        Args:
            ids: Айди изображений для обработки.
        """
        logger.info(f"Starting sync pipeline for {len(ids)} images (engine={self.engine})")  # noqa E501
        gen1 = self.download_generator(ids)
        gen2 = self.processing_generator(gen1)
        for msg in self.save_generator(gen2):
            logger.debug(msg)

        logger.info("Sync pipeline completed")


class AsyncImagePipeline:
    """Асинхронный пайплайн: скачивание -> обработка -> сохранение."""
    def __init__(self, save_dir: str = DEFAULT_SAVE_DIR, engine: str = "python",
                 operations: Optional[Dict[OperationType, Dict[str, Any]]] = None,
                 ) -> None:
        """
        Инициализация.

        Args:
            save_dir (str): Путь к папке для сохранения изображений.
            engine (str): Способ обработки (python или c).
            operations: Операции, которые нужно проделать с изображением.
        """
        self.save_dir = save_dir
        self.engine = engine
        self.operations = default_operations if not operations else operations
        os.makedirs(save_dir, exist_ok=True)

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

            results: Dict[str, np.ndarray] = {}

            for op_type, params in self.operations.items():
                if op_type == OperationType.GREYSCALE:
                    results["greyscale"] = artwork.color_to_greyscale()
                elif op_type == OperationType.BLUR:
                    results["blur"] = artwork.gaussian_blur(**params)
                elif op_type == OperationType.SOBEL:
                    results["sobel"] = artwork.sobel_transformation()
                elif op_type == OperationType.GAMMA:
                    results["gamma"] = artwork.gamma_correction(**params)
                elif op_type == OperationType.HISTOGRAM:
                    results["histogram"] = artwork.histogram_alignment()

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
