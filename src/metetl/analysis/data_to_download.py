"""Подготовка списка изображений для скачивания."""
import csv
import json

from ..logging_config import logger


def prepare_download_list(csv_path: str = "data/MetObjects.csv",
                          output_path: str = "data/to_download.json",
                          classification: str = "Paintings") -> list:
    """
    Читает CSV, отбирает строки с заданной классификацией и сохраняет их Object ID в JSON.

    Args:
        csv_path: Путь до датасета.
        output_path: Папка для сохранения изображений.
        classification: Тип изображений (пример: Paintings).

    Returns:
        ids: Список айди.

    Raises:
        Exception: Ошибка при чтении датасета.
    """
    logger.info(f"Preparing download list from {csv_path}")
    ids = []
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Classification') == classification:
                    ids.append(int(row['Object ID']))
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ids, f, indent=2)

    logger.info(f"Saved {len(ids)} IDs to {output_path}")
    return ids
