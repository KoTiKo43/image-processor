"""main файл"""
import asyncio
import csv
import random
import sys

import aiofiles

import aiohttp

from async_processor import AsyncImageDownloader

import numpy as np


DATASET = "MetObjects.csv"


def get_random_id_paintings(dataset_path: str = DATASET,
                            num: int = 1) -> np.ndarray[int]:
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

    return np.array(random.sample(painting_ids, min(num, len(painting_ids))))


async def download_all(ids: np.ndarray[int], downloader: AsyncImageDownloader) -> list:
    """
    Запускает скачивание изображений в асинхронном режиме.

    Args:
        ids: Список с айди изображений.
        downloader: Класс для асинхронного скачивания.

    Returns:
        list: Результаты скачивания.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [downloader.download_one(session, obj_id, i + 1)
                 for i, obj_id in enumerate(ids)]
        results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


def main() -> None:
    """Скачивание, обработка и сохранение заданного количество изображений асинхронно"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <number of images>")
        sys.exit(1)

    num_images = int(sys.argv[1])

    images_ids = get_random_id_paintings(DATASET, num_images)

    downloader = AsyncImageDownloader(save_dir="paintings")

    results = asyncio.run(download_all(images_ids, downloader))

    print(f"\nУспешно скачано {len(results)} изображений:")
    for idx, obj_id, img_path, json_path in results:
        print(f"  [{idx}] ID {obj_id}: {img_path}, {json_path}")


if __name__ == "__main__":
    main()
