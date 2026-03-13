"""Пример работы классов."""
from artwork import Artwork

import cv2

from image_processor import ImageProcessor, OperationType

DATASET = "MetObjects.csv"


def main() -> None:
    """Скачивание, обработка и сохранения изображения."""
    processor = ImageProcessor()

    artwork = processor.download_random_image(dataset_path=DATASET, small_image=True)

    ops = {
        OperationType.GREYSCALE: {},
        OperationType.BLUR: {"size": 7, "sigma": 2.0},
        OperationType.SOBEL: {},
        OperationType.GAMMA: {"gamma": 0.7},
        OperationType.HISTOGRAM: {},
    }

    results = processor.process_artwork(artwork, ops)

    for op_name, img in results.items():
        cv2.imwrite(f"{processor.save_dir}/{artwork.metadata['objectID']}_{op_name}.jpg",
                    img)


if __name__ == '__main__':
    main()
