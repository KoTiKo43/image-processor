"""Описание абстрактного класса Artwork
и его производных (обработка цветного и чёрно-белого изображений)."""
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class Artwork(ABC):
    """
    Абстрактный класс изображений.

    Attributes:
        image (ndarray): Изображение (доступно только для чтения).
            Для цветного изображения - трёхмерный массив (H, W, 3) в формате BGR,
            для чёрно-белого - двумерный (H, W).
        metadata (Dict): Метаданные объекта, полученные из API Метрополитен-музея.
            Содержит информацию о картине (название, автор, дата и т.д.).
    """

    __slots__ = ("_image", "_metadata")

    def __init__(self, image: np.ndarray, metadata: Dict) -> None:
        """
        Определение изображения и его метаданных.

        Args:
            image (ndarray): Массив изображения.
            metadata (Dict): Словарь метаданных.
        """
        self._image = image
        self._metadata = metadata

    @property
    def image(self) -> np.ndarray:
        """Доступ к изображению только для чтения."""
        return self._image

    @property
    def metadata(self) -> Dict:
        """Доступ к метаданным изображения только для чтения."""
        return self._metadata

    @abstractmethod
    def color_to_greyscale(self) -> np.ndarray:
        """
        Перевод цветного изображения в чёрно-белое.
        Должен возвращать 2D массив.
        """
        pass

    @abstractmethod
    def convolve2d(self, kernel: np.ndarray) -> np.ndarray:
        """
        Производит двумерную свёртку изображения.

        Args:
            kernel (ndarray): Маска.
        """
        pass

    @abstractmethod
    def gaussian_blur(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """
        Размытие изображения с помощью ядра Гаусса.

        Args:
            size (int): Размер ядра Гаусса.
            sigma (float): Стандартное среднее отклонение.
        """
        pass

    @abstractmethod
    def sobel_transformation(self) -> np.ndarray:
        """Осуществляет выделение границ с помощью оператора Собеля."""
        pass

    @abstractmethod
    def gamma_correction(self, gamma: float = 1.0) -> np.ndarray:
        """
        Производит гамма-коррекцию изображения по введённому значению гаммы.

        Args:
            gamma (float): Гамма для преобразования.
        """
        pass

    @abstractmethod
    def histogram_alignment(self) -> np.ndarray:
        """Производит выравнивание гистограммы изображения."""
        pass

    def __add__(self, other: 'Artwork') -> 'Artwork':
        """
        Перегрузка метода сложения изображений.

        Args:
            other (Artwork): Второе слагаемое.

        Raises:
            ValueError: Изображения должны быть одной формы (shape).

        Returns:
            Artwork: Сумма изображений, обрезанная в диапазоне 0-255.
        """
        if self._image.shape != other._image.shape:
            raise ValueError('Images must be the same shape!')

        new_image = np.clip(self._image + other._image, 0, 255).astype(np.uint8)
        return self.__class__(new_image, {})  # Метаданные не объединяем

    def __str__(self) -> str:
        """
        Перегрузка метода преобразования в строку.

        Returns:
            str: Строка с информацией об изображении.
        """
        return f'{self.__class__.__name__}: {self._metadata.get('title', 'Unknown')}, shape={self._image.shape}'  # noqa: E501
