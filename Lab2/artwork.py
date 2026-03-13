"""Описание абстрактного класса Artwork
и его производных (обработка цветного и чёрно-белого изображений)."""
from abc import ABC, abstractmethod
from typing import Dict

import cv2

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

    Args:
        image (ndarray): Исходное изображение.
        metadata (Dict): Словарь метаданных.

    Raises:
        ValueError: Если переданное изображение не соответствует заданной размерности.
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
        """Доступ к изображению (только для чтения)."""
        return self._image

    @property
    def metadata(self) -> Dict:
        """Доступ к метаданным изображения (только для чтения)."""
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

    @staticmethod
    def create(image: np.ndarray, metadata: dict) -> 'Artwork':
        """
        Фабричный метод: создаёт экземпляр подходящего подкласса Artwork
        в зависимости от размерности изображения.

        Args:
            image (npdarray): Массив изображения.
            metadata (Dict): Словарь метаданных.

        Raises:
            ValueError: Массив должен подходить под формат изображения.

        Returns:
            Объект для цветного или чёрно-белого изображения.
        """
        if image.ndim == 3 and image.shape[2] == 3:
            return ArtworkColor(image, metadata)
        elif image.ndim == 2:
            return ArtworkBW(image, metadata)
        else:
            raise ValueError("Unsupported image shape: expected 2D (greyscale) or 3D (color) array")  # noqa: E501

    def __add__(self, other: 'Artwork') -> 'Artwork':
        """
        Перегрузка метода сложения изображений.

        Args:
            other (Artwork): Второе слагаемое.

        Raises:
            ValueError: Если изображения разной формы (shape).

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


class ArtworkBW(Artwork):
    """Реализация для чёрно-белого изображения."""
    __slots__ = ()

    def __init__(self, image: np.ndarray, metadata: Dict) -> None:
        """
        Определение серого изображения, его метаданных.

        Args:
            image (ndarray): Массив изображения.
            metadata (Dict): Словарь метаданных.

        Raises:
            ValueError: Если изображение цветное, создать класс не получится.
        """
        if image.ndim != 2:
            raise ValueError("ArtworkBW requires 2D greyscale image")

        super().__init__(image, metadata)

    def color_to_greyscale(self) -> np.ndarray:
        """
        Изображение уже в градациях серого, метод возвращает копию.

        Returns:
            ndarray: Копия изображения в градациях серого.
        """
        return self._image.copy()

    def convolve2d(self, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет двумерную свёртку серого изображения на основе приведённой маски (ядра).

        Args:
            kernel (ndarray): Маска.

        Returns:
            result (ndarray): Массив с применённой маской.
        """
        from numpy.lib.stride_tricks import sliding_window_view
        kerh, kerw = kernel.shape

        # Отступы для Zero padding
        pad_h, pad_w = kerh // 2, kerw // 2
        # Матрица с отступами
        padded = np.pad(self._image, pad_width=((pad_h, pad_h), (pad_w, pad_w)),
                        mode='constant')

        windows = sliding_window_view(padded, (kerh, kerw))

        result = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))

        return result

    def gaussian_blur(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """
        Применяет гауссовское размытие при помощи двумерной свёртки.

        Args:
            size (int): Размер ядра Гаусса.
            sigma (float): Стандартное среднее отклонение.

        Returns:
            ndarray: Свёртка изображения с использованием ядра Гаусса.
        """
        kernel = self._gaussian_kernel(size, sigma)
        return self.convolve2d(kernel)

    def sobel_transformation(self) -> np.ndarray:
        """
        Осуществляет выделение границ с помощью оператора Собеля.

        Returns:
            g_matrix (ndarray): Обработанное с помощью оператора Собеля изображение.
        """
        gy = np.array([(1, 2, 1),
                       (0, 0, 0),
                       (-1, -2, -1)], dtype=np.float32)
        gx = np.array([(-1, 0, 1),
                       (-2, 0, 2),
                       (-1, 0, 1)], dtype=np.float32)

        grad_x = self.convolve2d(gx)
        grad_y = self.convolve2d(gy)

        g_matrix = np.sqrt(grad_x**2 + grad_y**2)
        # Нормализация в диапазон 0-255
        g_matrix = (g_matrix / g_matrix.max() * 255).astype(np.uint8)
        return g_matrix

    def gamma_correction(self, gamma: float = 1.0) -> np.ndarray:
        """
        Производит гамма-коррекцию изображения по введённому значению гаммы.

        Args:
            gamma (float): Гамма для преобразования.

        Returns:
            result (ndarray): Матрица обработанного изображения.
        """
        normalized_img = self._image.astype(np.float32) / 255.0
        gamma_img = normalized_img ** gamma
        result = (gamma_img * 255).astype(np.uint8)

        return result

    def histogram_alignment(self) -> np.ndarray:
        """
        Производит выравнивание гистограммы серого изображения.
        С созданием LUT (Look-Up Table).

        Returns:
            result (ndarray): Выравненное изображение.
        """
        hist, bins = np.histogram(self._image.flatten(), bins=256, range=[0, 256])
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        lut = cdf_normalized.astype(np.uint8)
        result = lut[self._image]
        return result

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """
        Создание Гауссовского ядра размытия.

        Args:
            size (int): Размерность ядра.
            sigma (float): Стандартное отклоение.

        Returns:
            kernel (np.ndarray): Ядро Гаусса.
        """
        # Создаём сетку координат
        ax, ay = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        kernel = np.exp(-(ax**2 + ay**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel


class ArtworkColor(Artwork):
    """
    Реализация для цветного изображения.

    Attributes:
        is_color (bool): Флаг цветности изображения.
    """
    __slots__ = ('_is_color',)

    def __init__(self, image: np.ndarray, metadata: Dict) -> None:
        """
        Определение цветного изображения, его метаданных.
        Установка флага цветности.

        Args:
            image (ndarray): Массив изображения.
            metadata (Dict): Словарь метаданных.

        Raises:
            ValueError: Если изображение нецветное, создать класс не получится.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("ArtworkColor requires 3-channel BGR image")

        super().__init__(image, metadata)
        self._is_color = True

    @property
    def is_color(self) -> bool:
        """Доступ к флагу цветности (только для чтения)."""
        return self._is_color

    def color_to_greyscale(self) -> np.ndarray:
        """
        Переводит цветное изображение в градации серого.

        Returns:
            greyscale (ndarray): Изображение в градациях серого.
        """
        weights = np.array([0.114, 0.587, 0.2989])  # Веса для преобразования
        greyscale = np.dot(self._image, weights).astype(np.uint8)
        return greyscale

    def convolve2d(self, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет двухмерную свёртку на цветное изображение (на каждый из каналов).

        Args:
            kernel (ndarray): Маска.

        Returns:
            img_conv (ndarray): Массив с применённой маской.
        """
        from numpy.lib.stride_tricks import sliding_window_view
        kerh, kerw = kernel.shape

        # Отступы для Zero padding
        pad_h, pad_w = kerh // 2, kerw // 2
        result = np.zeros_like(self._image, dtype=np.float32)
        # Обработка каналов по-отдельности
        for c in range(3):
            channel = self._image[..., c]
            padded = np.pad(channel, pad_width=((pad_h, pad_h), (pad_w, pad_w)),
                            mode='constant')
            windows = sliding_window_view(padded, (kerh, kerw))
            result[..., c] = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))

        return result

    def gaussian_blur(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """
        Применяет гауссовское размытие при помощи двумерной свёртки.

        Args:
            size (int): Размер ядра Гаусса.
            sigma (float): Стандартное среднее отклонение.

        Returns:
            ndarray: Свёртка изображения с использованием ядра Гаусса.
        """
        kernel = self._gaussian_kernel(size, sigma)
        return self.convolve2d(kernel)

    def sobel_transformation(self) -> np.ndarray:
        """
        Осуществляет выделение границ с помощью оператора Собеля.

        Returns:
            g_matrix (ndarray): Обработанное с помощью оператора Собеля изображение.
        """
        gy = np.array([(1, 2, 1),
                       (0, 0, 0),
                       (-1, -2, -1)], dtype=np.float32)
        gx = np.array([(-1, 0, 1),
                       (-2, 0, 2),
                       (-1, 0, 1)], dtype=np.float32)

        grad_x = self.convolve2d(gx)
        grad_y = self.convolve2d(gy)

        g_matrix = np.sqrt(grad_x**2 + grad_y**2)
        # Нормализация в диапазон 0-255
        g_matrix = (g_matrix / g_matrix.max() * 255).astype(np.uint8)
        return g_matrix

    def gamma_correction(self, gamma: float = 1.0) -> np.ndarray:
        """
        Производит гамма-коррекцию изображения по введённому значению гаммы.

        Args:
            gamma (float): Гамма для преобразования.

        Returns:
            result (ndarray): Матрица обработанного изображения.
        """
        normalized_img = self._image.astype(np.float32) / 255.0
        gamma_img = normalized_img ** gamma
        result = (gamma_img * 255).astype(np.uint8)

        return result

    def histogram_alignment(self) -> np.ndarray:
        """
        Производит выравнивание гистограммы цветного изображения.
        С созданием LUT (Look-Up Table).

        Returns:
            result (ndarray): Матрица выравненного изорабжения (BGR или одноканал).
        """
        img_lab = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)
        l_brightneses, a_color, b_color = cv2.split(img_lab)

        hist, bins = np.histogram(l_brightneses, bins=256, range=[0, 256])
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        lut = cdf_normalized.astype(np.uint8)
        l_result = lut[l_brightneses]

        result_lab = cv2.merge([l_result, a_color, b_color])
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        return result

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """
        Создание Гауссовского ядра размытия.

        Args:
            size (int): Размерность ядра.
            sigma (float): Стандартное отклоение.

        Returns:
            kernel (np.ndarray): Ядро Гаусса.
        """
        # Создаём сетку координат
        ax, ay = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        kernel = np.exp(-(ax**2 + ay**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel
