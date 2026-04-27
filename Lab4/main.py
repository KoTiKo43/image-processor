"""main файл"""
import asyncio
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import aiohttp

from artwork import Artwork

from async_processor import AsyncImageDownloader

import cv2


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


async def download_all(ids: list[int], downloader: AsyncImageDownloader) -> list:
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


def process_image_task(img_path: str, json_path: str, index: int,
                       obj_id: int, save_dir: str) -> int:
    """
    Функция, которая будет запускаться в отдельном процессе.

    Args:
        img_path: Путь к оригинальному изображению.
        json_path: Путь к метаданным изображения.
        index: Индекс изображения.
        obj_id: ID изображения.
        save_dir: Папка для сохранения обработанных изображений.

    Returns:
        index: Индекс изображения
    """
    print(f"[PROCESS][PID {os.getpid()}] Image {index} (ID: {obj_id}) started")

    image = cv2.imread(img_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    artwork = Artwork.create(image, metadata)

    processed_dir = os.path.join(save_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    grey = artwork.color_to_greyscale()
    blurred = artwork.gaussian_blur(size=5, sigma=1.0)
    sobel = artwork.sobel_transformation()
    gamma = artwork.gamma_correction(gamma=1.5)
    histogram = artwork.histogram_alignment()

    modes = {
        'greyscale': grey,
        'blur': blurred,
        'sobel': sobel,
        'gamma': gamma,
        'histogram': histogram,
    }

    for mode, image in modes.items():
        fname = f"{index}_{obj_id}_{mode}.jpg"
        fpath = os.path.join(processed_dir, fname)
        cv2.imwrite(fpath, image)
        print(f"[PROCESS][PID {os.getpid()}] Image {index} (ID: {obj_id}) {mode} saved")

    print(f"[PROCESS][PID {os.getpid()}] Image {index} (ID: {obj_id}) finished")
    return index


def run_processing(download_results: tuple, save_dir: str) -> None:
    """
    Функция запускает мультипроцессорную обработку изображений.

    Args:
        download_results: index, obj_id, img_path, json_path.
        save_dir: Папка для сохранения обработанных изображений.
    """
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for index, obj_id, img_path, json_path in download_results:
            future = executor.submit(
                process_image_task, img_path, json_path, index, obj_id, save_dir)
            futures.append(future)

        for future in futures:
            future.result()


def main() -> None:
    """Скачивание, обработка и сохранение заданного количество изображений асинхронно"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <number of images>")
        sys.exit(1)

    num_images = int(sys.argv[1])

    images_ids = get_random_id_paintings(DATASET, num_images)

    downloader = AsyncImageDownloader(save_dir="paintings")

    start_time = time.perf_counter()

    results = asyncio.run(download_all(images_ids, downloader))

    download_time = time.perf_counter() - start_time
    print(f"[TIME] Download phase took {download_time:.2f} seconds")

    proc_start = time.perf_counter()
    run_processing(results, save_dir="paintings")
    proc_time = time.perf_counter() - proc_start
    print(f"[TIME] Processing phase took {proc_time:.2f} seconds")
    print(f"[TIME] Total time: {time.perf_counter() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
