"""Асинхронная обработка изображений"""
import asyncio
import csv
import json
import os
import time

import aiofiles

import aiohttp

from artwork import Artwork

import numpy as np


API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1/objects"


class AsyncImageDownloader:
    """"""
    def __init__(self, save_dir: str = "paintings") -> None:
        """
        Инициализация save_dir

        Args:
            save_dir (str): Путь к папке для сохранения изображений.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    async def fetch_json(self, session: aiohttp.ClientSession, object_id: int) -> dict:
        """
        Асинхронно подтягивает json объекта по ссылке из API_BASE

        Args:
            session: Асинхронная клиентская сессия.
            object_id: id нужного объекта.

        Returns:
            dict: json объекта
        """
        url = f'{API_BASE}/{object_id}'
        async with session.get(url) as resp:
            return await resp.json()

    async def download_image(self, session: aiohttp.ClientSession, url: str,
                             filepath: str) -> None:
        """
        Асинхронно скачивает изображение по пути filepath

        Args:
            session: Асинхронная клиентская сессия.
            url: Ссылка на изображение.
            filepath: Путь для сохранения изображения.
        """
        async with session.get(url) as resp:
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(await resp.read())

    async def download_one(self, session: aiohttp.ClientSession, object_id: str,
                           index: int) -> tuple[int, int, str, str]:
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
        print(f"[DOWNLOAD] Image {index} (ID: {object_id}) started")
        painting_json = await self.fetch_json(session, object_id)
        if not (painting_json.get('isPublicDomain') and painting_json.get('primaryImage')):
            print(f"[DOWNLOAD] Image {index} (ID: {object_id}) skipped (not public domain)")
            return None

        original_dir = os.path.join(self.save_dir, "original")
        os.makedirs(original_dir, exist_ok=True)

        painting_path = os.path.join(original_dir, f"{index}_{object_id}_original.jpg")
        json_path = os.path.join(original_dir, f"{index}_{object_id}_metadata.json")

        async with aiofiles.open(json_path, 'w', encoding="utf-8") as jf:
            await jf.write(json.dumps(painting_json, indent=2, ensure_ascii=False))

        painting_url = painting_json.get('primaryImage')
        await self.download_image(session, painting_url, painting_path)

        print(f"[DOWNLOAD] Image {index} (ID: {object_id}) finished -> {painting_path}")
        return index, object_id, painting_path, json_path
