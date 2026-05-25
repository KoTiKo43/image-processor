"""Декораторы
timer
async_timer
"""
import time
from functools import wraps
from typing import Callable

from metetl.logging_config import logger


def timer(func: Callable) -> Callable:
    """
    Декоратор. Засекает время выполнения функции.

    Args:
        func (Callable): Функция для оборачивания.

    Returns:
        wrapper (Callable): Обёрнутая функция (с таймером).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):                           # noqa: ANN002, ANN003, ANN201
        """
        Оборачивает функцию таймером.

        Args:
            args: Аргументы оборачиваемой функции (распаковываются).
            kwargs: Аргументы оборачиваемой функции (распаковываются).

        Returns:
            result: Результат выполнения функции без обёртки.
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f'Function {func.__name__} executed in {end - start:.4f} seconds')
        return result

    return wrapper


def async_timer(func: Callable) -> Callable:
    """
    Декоратор. Засекает время выполнения асинхронной функции.

    Args:
        func (Callable): Функция для оборачивания.

    Returns:
        wrapper (Callable): Обёрнутая функция (с таймером).
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):                     # noqa: ANN002, ANN003, ANN201
        """
        Оборачивает асинхронную функцию таймером.

        Args:
            args: Аргументы оборачиваемой функции (распаковываются).
            kwargs: Аргументы оборачиваемой функции (распаковываются).

        Returns:
            result: Результат выполнения функции без обёртки.
        """
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f'Async function {func.__name__} executed in {end - start:.4f} seconds')  # noqa E501
        return result

    return wrapper
