"""
Скачивает и обрабатывает изображение.
"""

import download_painting

import process_painting


def main() -> None:
    """
    Скачивает и обрабатывает изображение.
    """
    download_painting.main()
    process_painting.main()


if __name__ == "__main__":
    main()
