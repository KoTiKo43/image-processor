"""Функция и экземляр логгера"""
import logging
import sys


def setup_logging(log_file: str = "logs/metetl.log") -> logging.Logger:
    """
    Создаёт и настраивает главный логгер проекта.

    Args:
        log_file: Путь и название файла для сохранения логов.

    Returns:
        logger: Экземляр логгера.
    """
    logger = logging.getLogger("metetl")
    logger.setLevel("DEBUG")

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel("DEBUG")
    f_formatter = logging.Formatter(
        '{asctime} - {name} - {levelname} - {filename}:{lineno} - {message}',
        style='{',
        datefmt='%Y-%m-%d %H:%M',
    )
    fh.setFormatter(f_formatter)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    c_formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(c_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Глобальный экземпляр логгера
logger = setup_logging()
