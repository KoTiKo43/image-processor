"""
Обрабатывает изображение согласно реализованным операциям обработки.
Реализованы:
1. Представление в градациях серого (color_to_greyscale)
2. Двумерная свёртка (convolve2d)
3. Гауссовское размытие (gaussian_blur)
4. Выделение границ с помощью оператора Собеля (sobel_transformation)
5. Гамма-коррекция (gamma_correction)
6. Выравнивание гистограммы изображения (histogram_alignment)
"""

import os
import time

import cv2

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PAINTING_DIR = "paintings"


def color_to_greyscale(img: np.ndarray) -> np.ndarray:
    """
    Преобразовывает цветную картинку в градации серого.

    Args:
        img (ndarray): Матрица изображения (BGR).

    Returns:
        greyscale (ndarray): Матрица изображения в градациях серого.
    """
    weights = np.array([0.114, 0.587, 0.2989])  # Веса для преобразования
    # Проверяем, что изображение цветное
    if img.ndim == 3 and img.shape[2] == 3:
        greyscale = np.dot(img, weights).astype(np.uint8)
    else:
        greyscale = img

    return greyscale


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Выполняет двухмерную свёртку на цветное изображение (на каждый из каналов).

    Args:
        img (ndarray): Массив изображения (BGR или одноканал).
        kernel (ndarray): Маска.

    Returns:
        img_conv (ndarray): Массив с применённой маской.
    """
    def convolve2d_greyscale(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет двумерную свёртку изображения на основе приведённой маски (ядра).

        Args:
            img (ndarray): Канал изображения.
            kernel (ndarray): Маска.

        Returns:
            output (ndarray): Массив с применённой маской.
        """
        imgh, imgw = img.shape
        kerh, kerw = kernel.shape

        # Отступы для Zero padding
        pad_h, pad_w = kerh // 2, kerw // 2
        # Матрица с отступами
        padded = np.pad(img, pad_width=((pad_h, pad_h), (pad_w, pad_w)),
                        mode='constant')

        windows = sliding_window_view(padded, (kerh, kerw))

        result = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))

        return result

    if img.ndim == 3 and img.shape[2] == 3:
        blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        blue_conv = convolve2d_greyscale(blue, kernel)
        green_conv = convolve2d_greyscale(green, kernel)
        red_conv = convolve2d_greyscale(red, kernel)
        img_conv = cv2.merge([blue_conv, green_conv, red_conv])
    else:
        img_conv = convolve2d_greyscale(img, kernel)

    return img_conv


def gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """
    Создание Гауссовского ядра размытия.

    Args:
        size (int): Размерность ядра.
        sigma (float): Стандартное отклоение.

    Returns:
        kernel (np.ndarray): Ядро Гаусса.
    """
    # Создаём сетку координат
    ax, ay = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    kernel = np.exp(-(ax**2 + ay**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur(img: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Применяет гауссовское размытие при помощи двумерной свёртки.

    Args:
        img (ndarray): Матрица изображения (BGR или одноканал).
        size (int): Размер ядра Гаусса.
        sigma (float): Стандартное среднее отклонение.

    Returns:
        ndarray: Свёртка изображения с использованием ядра Гаусса.
    """
    kernel = gaussian_kernel(size, sigma)

    return convolve2d(img, kernel)


def sobel_transformation(img: np.ndarray) -> np.ndarray:
    """
    Осуществляет выделение границ с помощью оператора Собеля.

    Args:
        img (ndarray): Матрица изображения (BGR или одноканал).

    Returns:
        g (ndarray): Обработанное с помощью оператора Собеля изображение.
    """
    gy = np.array([(1, 2, 1),
                   (0, 0, 0),
                   (-1, -2, -1)], dtype=np.float32)
    gx = np.array([(-1, 0, 1),
                   (-2, 0, 2),
                   (-1, 0, 1)], dtype=np.float32)

    grad_x = convolve2d(img, gx)
    grad_y = convolve2d(img, gy)

    g_matrix = np.sqrt(grad_x**2 + grad_y**2)
    # Нормализация в диапазон 0-255
    g_matrix = (g_matrix / g_matrix.max() * 255).astype(np.uint8)
    return g_matrix


def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Производит гамма-коррекцию изображения по введённому значению гаммы.

    Args:
        img (ndarray): Матрица изображения (BGR или одноканал).
        gamma (float): Гамма для преобразования.

    Returns:
        result (ndarray): Матрица обработанного изображения.
    """
    normalized_img = img.astype(np.float32) / 255.0
    gamma_img = normalized_img ** gamma
    result = (gamma_img * 255).astype(np.uint8)

    return result


def histogram_alignment(img: np.ndarray) -> np.ndarray:
    """
    Производит выравнивание гистограммы цветного или серого изображения.

    Args:
        img (ndarray): Матрица изображения (BGR или одноканал).

    Returns:
        result (ndarray): Матрица выравненного изорабжения (BGR или одноканал).
    """
    def histogram_alignment_greyscale(img_greyscale: np.ndarray) -> np.ndarray:
        """
        Производит выравнивание гистограммы серого изображения.
        С созданием LUT (Look-Up Table)

        Args:
            img_greyscale (ndarray): Матрица изображения (одноканал).

        Returns:
            result (ndarray): Выравненное изображение.
        """
        hist, bins = np.histogram(img_greyscale.flatten(), bins=256, range=[0, 256])
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        lut = cdf_normalized.astype(np.uint8)
        result = lut[img_greyscale]
        return result

    if img.ndim == 3 and img.shape[2] == 3:
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_brightneses, a_color, b_color = cv2.split(img_lab)
        l_result = histogram_alignment_greyscale(l_brightneses)
        result_lab = cv2.merge([l_result, a_color, b_color])
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    else:
        result = histogram_alignment_greyscale(img)

    return result


def image_name_addition(image_path: str, addition: str) -> str:
    """
    Добавляет текст к названию изображения.

    Args:
        image_path (str): Путь до изображения.
        addition (str): Что требуется добавить к названию изображения.

    Returns:
        str: Новое название
    """
    return image_path[:-4] + f'_{addition}.jpg'


def main() -> None:
    """
    Выполнение функций и сравнение их с библиотечными.
    """
    # Список изображений картин
    images = os.listdir(PAINTING_DIR)
    # Список изображений с путями
    images = [os.path.join(PAINTING_DIR, image) for image in images
              if os.path.join(PAINTING_DIR, image)[-4:] != 'json']
    # Оставляем только файлы
    images = [image for image in images if os.path.isfile(image)]
    # Берём последнее добавленное изображение
    image_path = max(images, key=os.path.getctime)

    image = cv2.imread(image_path)

    # Создание полутонового (в градациях серого) изображения
    start = time.perf_counter()
    image_greyscale = color_to_greyscale(image)
    t_greyscale_custom = time.perf_counter() - start

    start = time.perf_counter()
    image_greyscale_opencv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t_greyscale_opencv = time.perf_counter() - start

    image_greyscale_path = image_name_addition(image_path, 'greyscale')
    image_greyscale_opencv_path = image_name_addition(image_path, 'greyscale_opencv')
    cv2.imwrite(image_greyscale_path, image_greyscale)
    cv2.imwrite(image_greyscale_opencv_path, image_greyscale_opencv)
    print(f"Greyscale: custom = {t_greyscale_custom}s, opencv = {t_greyscale_opencv}s")

    # Двумерная свёртка (повышение резкости)
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)

    start = time.perf_counter()
    image_convolve2d = convolve2d(image, kernel_sharpen)
    t_convolve2d_custom = time.perf_counter() - start

    start = time.perf_counter()
    image_convolve2d_opencv = cv2.filter2D(image, -1, kernel_sharpen)
    t_convolve2d_opencv = time.perf_counter() - start

    image_convolve2d_path = image_name_addition(image_path, 'convolve2d')
    image_convolve2d_opencv_path = image_name_addition(image_path,
                                                       'convolve2d_opencv')
    cv2.imwrite(image_convolve2d_path, image_convolve2d)
    cv2.imwrite(image_convolve2d_opencv_path, image_convolve2d_opencv)
    print(f"Convolve2D: custom = {t_convolve2d_custom}s, opencv = {t_convolve2d_opencv}s")

    # Гауссовское сглаживание изображения с помощью двумерной свёртки
    start = time.perf_counter()
    image_blur = gaussian_blur(image)
    t_blur_custom = time.perf_counter() - start

    start = time.perf_counter()
    image_blur_opencv = cv2.GaussianBlur(image, (5, 5), 1.0)
    t_blur_opencv = time.perf_counter() - start

    image_blur_path = image_name_addition(image_path, 'blur')
    image_blur_opencv_path = image_name_addition(image_path, 'blur_opencv')
    cv2.imwrite(image_blur_path, image_blur)
    cv2.imwrite(image_blur_opencv_path, image_blur_opencv)
    print(f"Blur: custom = {t_blur_custom}s, opencv = {t_blur_opencv}")

    # Выделение границ с помощью оператора Собеля
    start = time.perf_counter()
    image_sobel = sobel_transformation(image)
    t_sobel_custom = time.perf_counter() - start

    start = time.perf_counter()
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    image_sobel_opencv = np.sqrt(sobelx**2 + sobely**2)
    image_sobel_opencv = (image_sobel_opencv /
                          image_sobel_opencv.max() * 255).astype(np.uint8)
    t_sobel_opencv = time.perf_counter() - start

    image_sobel_path = image_name_addition(image_path, 'sobel')
    image_sobel_path_opencv = image_name_addition(image_path, 'sobel_opencv')
    cv2.imwrite(image_sobel_path, image_sobel)
    cv2.imwrite(image_sobel_path_opencv, image_sobel_opencv)
    print(f"Sobel: custom = {t_sobel_custom}s, opencv = {t_sobel_opencv}s")

    # Применение гамма-коррекции
    image_gamma = gamma_correction(image, 0.5)

    image_gamma_path = image_name_addition(image_path, 'gamma')
    cv2.imwrite(image_gamma_path, image_gamma)

    # Применение выравнивания гистограммы
    image_histogram = histogram_alignment(image)

    image_histogram_path = image_name_addition(image_path, 'histogram')
    cv2.imwrite(image_histogram_path, image_histogram)


if __name__ == '__main__':
    main()
