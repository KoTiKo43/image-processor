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
                   for obj_id, index in enumerate(ids, start=1)]

        for corout in asyncio.as_completed(corouts):
            result = await corout
            if result is None:
                continue

            index, obj_id, image_path, json_path = result

            image = await asyncio.to_thread(cv2.imread, image_path)
            async with aiofiles.open(json_path, 'r', encoding="utf-8") as jf:
                image_json = json.loads(await jf.read())

            yield index, obj_id, image, image_json


def process_image(image: np.ndarray, image_json: dict, index: int, obj_id: int,
                  ) -> Tuple[int, int, dict]:
    """
    Обработка изображения.

    Args:
        image: Массив изображения.
        image_json: Метаданные изображения.
        index: Порядковый номер изображения.
        obj_id: ID изображения.

    Returns:
        index, obj_id, results: Индекс, ID и словарь с результатами.
    """
    print(f"[PROCESS][PID {os.getpid()}] Image {index} (ID: {obj_id}) started")
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


async def process_image_pipeline(source: AsyncGenerator[Tuple[int, int, np.ndarray, dict]],
                                 save_dir: str,
                                 ) -> AsyncGenerator[Tuple[int, int, dict]]:
    """
    
    """
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for index, obj_id, img_path, json_path in source:
            future = executor.submit(
                process_image, img_path, json_path, index, obj_id, save_dir)
            futures.append(future)

        for completed in futures:
            index, obj_id, results = await completed
            yield index, obj_id, results
