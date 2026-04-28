"""Дополнительное задание: асинхронный пайплайн генераторов"""
import asyncio
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import AsyncGenerator, Tuple

import aiofiles

import aiohttp

from artwork import Artwork

from async_processor import AsyncImageDownloader

import cv2

import numpy as np


DATASET = "MetObjects.csv"


def get_random_id_paintings(dataset_path: str = DATASET,
                            num: int = 1) -> list[int]:
    """
    Скачивание случайных картин из датасета в заданном количестве.

    Args:
        dataset_path (str): Путь к датасету.
        num (int): Количество изображений для скачивания.

    Returns:
        list: Список с ID отобранных картин.
    """
    with open(dataset_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        painting_ids = [int(row['Object ID']) for row in reader
                        if row['Classification'] == 'Paintings']

    return random.sample(painting_ids, min(num, len(painting_ids)))


async def download_pipeline(ids: list[int], downloader: AsyncImageDownloader,
                            ) -> AsyncGenerator[Tuple[int, int, np.ndarray, dict]]:
    """
    Асинхронный генератор скачивания.

    Args:
        ids: ID изображений.
        downloader: Класс для асинхронной загрузки изображений.

    Yields:
        index, obj_id, image, image_json: Необходимые данные об изображении.
    """
    async with aiohttp.ClientSession() as session:
        corouts = [downloader.download_one(session, obj_id, index)
                   for index, obj_id in enumerate(ids, start=1)]

        for corout in asyncio.as_completed(corouts):
            result = await corout
            if result is not None:
                yield result


def process_image(image_path: str, image_json: dict, index: int, obj_id: int,
                  ) -> Tuple[int, int, dict]:
    """
    Обработка изображения.

    Args:
        image_path: Массив к изображению.
        image_json: Метаданные изображения.
        index: Порядковый номер изображения.
        obj_id: ID изображения.

    Returns:
        index, obj_id, results: Индекс, ID и словарь с результатами.
    """
    print(f"[PROCESS][PID {os.getpid()}] Image {index} (ID: {obj_id}) started")
    image = cv2.imread(image_path)
    artwork = Artwork.create(image, image_json)

    grey = artwork.color_to_greyscale()
    blurred = artwork.gaussian_blur(size=5, sigma=1.0)
    sobel = artwork.sobel_transformation()
    gamma = artwork.gamma_correction(gamma=1.5)
    histogram = artwork.histogram_alignment()

    results = {
        'greyscale': grey,
        'blur': blurred,
        'sobel': sobel,
        'gamma': gamma,
        'histogram': histogram,
    }

    print(f"[PROCESS][PID {os.getpid()}] Image {index} (ID: {obj_id}) finished")
    return index, obj_id, results


async def process_image_pipeline(source: AsyncGenerator[Tuple[int, int, np.ndarray, dict]],  # noqa E501
                                 ) -> AsyncGenerator[Tuple[int, int, dict]]:
    """
    Генератор обработки.

    Args:
        source: Асинхронный генератор с данными об изображении.

    Yields:
        item: Кортеж с индексом, ID объекта и словарём обработанных изображений.
    """
    result_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    background_tasks: list[asyncio.Task] = []

    async def wait_and_enqueue(future: asyncio.Future) -> None:
        """
        Ждёт завершения обработки и отправляет результат в очередь.

        Args:
            future: Объект Future.
        """
        res = await future
        await result_queue.put((res[0], res[1], res[2]))

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        async for index, obj_id, img_path, json_path in source:
            future = loop.run_in_executor(
                executor,
                process_image,
                img_path,
                json_path,
                index,
                obj_id,
            )
            task = asyncio.create_task(
                wait_and_enqueue(future)
            )
            background_tasks.append(task)

        if background_tasks:
            await asyncio.gather(*background_tasks)

        await result_queue.put(None)

    while True:
        item = await result_queue.get()
        if item is None:
            break

        yield item


async def save_pipeline(source: AsyncGenerator[Tuple[int, int, dict], None],
                        save_dir: str) -> AsyncGenerator[str, None]:
    """
    Генератор сохранения.
    Сохраняет файлы неблокирующе через asyncio.to_thread.

    Args:
        source: Асинхронный генератор с данными об изображении.
        save_dir: Папка сохранения.

    Yields:
        str: Строку о том, что файл сохранён.
    """
    processed_dir = os.path.join(save_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    async for index, obj_id, results in source:
        for mode, img in results.items():
            path = os.path.join(processed_dir, f"{index}_{obj_id}_{mode}.jpg")

            await asyncio.to_thread(cv2.imwrite, path, img)
            print(f"[SAVE] Image {index} (ID: {obj_id}) {mode} saved")

        yield f"Image {index} (ID: {obj_id}) fully saved"


async def run_pipeline(num_images: int, save_dir: str = "paintings") -> None:
    """
    Запускает пайплайн генераторов.

    Args:
        num_images: Количество изображений.
        save_dir: Путь для сохранения изображений.
    """
    ids = get_random_id_paintings(DATASET, num_images)
    downloader = AsyncImageDownloader(save_dir=save_dir)

    start = time.perf_counter()

    stage1 = download_pipeline(ids, downloader)
    stage2 = process_image_pipeline(stage1)
    stage3 = save_pipeline(stage2, save_dir)

    count = 0
    async for msg in stage3:
        print(f"[PIPELINE] {msg}")
        count += 1

    elapsed = time.perf_counter() - start
    print(f"[PIPELINE] Completed {count} images in {elapsed:.2f} seconds")


def main() -> None:
    """main функция"""
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <number of images>")
        sys.exit(1)

    num = int(sys.argv[1])
    asyncio.run(run_pipeline(num))


if __name__ == "__main__":
    main()
