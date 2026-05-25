"""Тест классов Artwork"""
import unittest

from metetl.images.models import ArtworkBW, ArtworkColor

import numpy as np


class TestArtworkBW(unittest.TestCase):
    def setUp(self):
        """Создаётся базовый чёрно-белый Artwork размером 10 на 10."""
        self.img = np.arange(100).reshape(10, 10).astype(np.uint8)
        self.artwork = ArtworkBW(self.img, {"title": "test"})

    def test_greyscale_returns_copy(self):
        """Проверяет, что в данном случае вернётся копия."""
        grey = self.artwork.color_to_greyscale()
        self.assertTrue(np.array_equal(grey, self.img))

        grey[0, 0] = 255
        self.assertNotEqual(self.artwork.image[0, 0], 255)

    def test_convolve2d(self):
        """Проверка того, что одиночнон ядро не изменит изображение."""
        kernel = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32)
        res = self.artwork.convolve2d(kernel)
        self.assertTrue(np.allclose(res, self.img.astype(np.float32)))

    def test_gaussian_blur_size(self):
        """Проверка равенства размерности  изображений до и после обработки."""
        blurred = self.artwork.gaussian_blur(size=5, sigma=1.0)
        self.assertEqual(blurred.shape, self.img.shape)


class TestArtworkColor(unittest.TestCase):
    def setUp(self):
        """Создаётся базовый цветной Artwork размером 10 на 10."""
        self.img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        self.artwork = ArtworkColor(self.img, {"title": "test"})

    def test_color_to_greyscale(self):
        """Проверка того, что в итоге получится чёрно-белое изображение."""
        grey = self.artwork.color_to_greyscale()
        self.assertEqual(len(grey.shape), 2)

    def test_sobel(self):
        """Проверка того, что при собеле вернётся цветное изображение."""
        sobel = self.artwork.sobel_transformation()
        self.assertEqual(sobel.shape, self.img.shape)
