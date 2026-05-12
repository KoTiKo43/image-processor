"""Тест SyncImagePipeline (ImageProcessor)"""
import json
import os
import unittest
from unittest.mock import ANY, MagicMock, call, mock_open, patch

from metetl.images.processing import ImagePipeline, OperationType

import numpy as np


class TestSyncImagePipelineIntegration(unittest.TestCase):
    def setUp(self):
        ops = {
            OperationType.GREYSCALE: {},
            OperationType.BLUR: {"size": 3, "sigma": 0.5},
        }
        self.pipeline = ImagePipeline(save_dir="test_images", engine="python", operations=ops)

    @patch("cv2.imwrite")
    @patch("metetl.images.processing.json.load")
    @patch("cv2.imread")
    @patch("builtins.open", new_callable=mock_open)
    @patch("metetl.images.processing.Artwork.create")
    @patch("metetl.images.processing.requests.get")
    def test_pipeline_uses_artwork_methods(self,
                                           mock_get, mock_create, mock_file,
                                           mock_imread, mock_json_load, mock_imwrite):
        fake_api_resp = MagicMock()
        fake_api_resp.json.return_value = {
            "isPublicDomain": True,
            "primaryImage": "http://example.com/image.jpg",
        }
        fake_img_resp = MagicMock()
        fake_img_resp.content = b"fake_image_data"
        mock_get.side_effect = [fake_api_resp, fake_img_resp]

        fake_metadata = {"title": "test", "objectID": 123}
        mock_json_load.return_value = fake_metadata

        fake_image = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_imread.return_value = fake_image

        mock_artwork = MagicMock()
        mock_artwork.color_to_greyscale.return_value = np.zeros((10, 10), dtype=np.uint8)
        mock_artwork.gaussian_blur.return_value = np.ones((10, 10), dtype=np.uint8)
        mock_create.return_value = mock_artwork

        ids = [123]
        self.pipeline.run_pipeline(ids)

        expected_api_calls = [
            call("https://collectionapi.metmuseum.org/public/collection/v1/objects/123"),
            call("http://example.com/image.jpg"),
        ]
        mock_get.assert_has_calls(expected_api_calls, any_order=False)

        expected_meta = os.path.join("test_images", "original", "1_123_metadata.json")
        mock_file.assert_any_call(expected_meta, "r")

        mock_create.assert_called_once_with(fake_image, fake_metadata)

        mock_artwork.color_to_greyscale.assert_called_once()
        mock_artwork.gaussian_blur.assert_called_once_with(size=3, sigma=0.5)

        expected_save_calls = [
            call(os.path.join("test_images", "processed", "1_123_greyscale.jpg"), ANY),
            call(os.path.join("test_images", "processed", "1_123_blur.jpg"), ANY),
        ]
        mock_imwrite.assert_has_calls(expected_save_calls, any_order=True)
        mock_imwrite.assert_has_calls(expected_save_calls, any_order=True)
        self.assertEqual(mock_imwrite.call_count, 2)
