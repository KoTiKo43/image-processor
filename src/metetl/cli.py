"""Модуль обработки консольного интерфейса."""
import argparse
import asyncio
import json
import os
import random
import sys

from metetl.analysis.aggregations import run_analysis
from metetl.analysis.data_to_download import prepare_download_list
from metetl.images.processing import AsyncImagePipeline, ImagePipeline, OperationType
from metetl.logging_config import logger


def main() -> None:
    """Настройка и работа парсера."""
    parser = argparse.ArgumentParser(prog="metetl", description="MET ETL Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    prepare_parser = subparsers.add_parser("prepare",
                                           help="Preparing json of paintings to download")
    prepare_parser.add_argument("--csv", default="data/MetObjects.csv",
                                help="Path to a csv file")
    prepare_parser.add_argument("--output", default="data/to_download.json",
                                help="Path to save data of paintings to download")

    process_parser = subparsers.add_parser("process",
                                           help="Processing random images")
    process_parser.add_argument("--input", default="data/to_download.json",
                                help="Path to json with paintings to download")
    process_parser.add_argument("--output", default="images",
                                help="Folder to save images")
    process_parser.add_argument("--num", type=int, default=1,
                                help="Amount of images to download and process")
    process_parser.add_argument("--engine", choices=["python", "c"], default="python",
                                help="Processing engine (python or c)")
    process_parser.add_argument("--async", action="store_true", dest="async_mode",
                                help="Starting ascynchronous pipeline")
    process_parser.add_argument("--operations", nargs="+",
                                default=["greyscale", "blur", "sobel", "gamma", "histogram"],  # noqa E501
                                help="List of operations to apply")
    process_parser.add_argument("--blur-size", type=int, default=5)
    process_parser.add_argument("--blur-sigma", type=float, default=1.0)
    process_parser.add_argument("--gamma-value", type=float, default=1.5)

    analyze_parser = subparsers.add_parser("analyze",
                                           help="Analyzing dataset")
    analyze_parser.add_argument("--csv", default="data/MetObjects.csv",
                                help="Path to a csv file")
    analyze_parser.add_argument("--output-dir", default="data/plots",
                                help="Path to a folder for saving analysis plots")

    args = parser.parse_args()

    if args.command == "prepare":
        try:
            prepare_download_list(args.csv, args.output)
            logger.info(f"Download list created: {args.output}")
        except Exception as e:
            logger.error(f"Creating download list failed: {e}")
            sys.exit(1)
    elif args.command == "process":
        ops = {}
        for op_name in args.operations:
            op_name = op_name.lower()
            if op_name == "greyscale":
                ops[OperationType.GREYSCALE] = {}
            elif op_name == "blur":
                ops[OperationType.BLUR] = {"size": args.blur_size, "sigma": args.blur_sigma}  # noqa E501
            elif op_name == "sobel":
                ops[OperationType.SOBEL] = {}
            elif op_name == "gamma":
                ops[OperationType.GAMMA] = {"gamma": args.gamma_value}
            elif op_name == "histogram":
                ops[OperationType.HISTOGRAM] = {}
            else:
                logger.warning(f"Unknown operation '{op_name}' ignored")

        try:
            with open(args.input, 'r') as f:
                ids = json.load(f)

            if not isinstance(ids, list):
                logger.error("Input JSON must to be a list of IDs")
                sys.exit(1)

            selected_ids = random.sample(ids, min(args.num, len(ids)))
            logger.info(f"Selected {len(selected_ids)} ids for processing")

            if args.async_mode:
                pipeline = AsyncImagePipeline(args.output, args.engine, operations=ops)
                asyncio.run(pipeline.run_pipeline(selected_ids))
            else:
                pipeline = ImagePipeline(args.output, args.engine, operations=ops)
                pipeline.run_pipeline(selected_ids)

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            sys.exit(1)

    elif args.command == "analyze":
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            run_analysis(args.output_dir, args.csv)
        except Exception as e:
            logger.error(f"Failed to analyze dataset: {e}")
            sys.exit(1)
    else:
        parser.print_help()
