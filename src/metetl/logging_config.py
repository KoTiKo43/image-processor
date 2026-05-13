"""Настройка логгера и его экземпляр"""
import logging.config
import os
import sys

import yaml


def setup_logging(config_path: str = None) -> logging.Logger:
    """
    Настраивает логирование из YAML-файла.
    Если путь не указан, ищет 'logging_config.yaml' рядом с этим модулем.

    Args:
        config_path: Путь и название файла для сохранения логов.

    Returns:
        logger: Экземляр логгера.
    """
    if config_path is None:
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, "logging_config.yaml")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    file_handler_config = config.get('handlers', {}).get('file', {})
    log_filename = file_handler_config.get('filename')
    if log_filename:
        log_dir = os.path.dirname(log_filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    logging.config.dictConfig(config)
    logger = logging.getLogger("metetl")

    return logger


logger = setup_logging()
