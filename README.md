# metetl - MET ETL Pipeline

Установка:
```
pip install .
```

Использование:
```
metetl prepare --csv data/MetObjects.csv --output data/to_download.json
metetl process --input data/to_download.json --output images --num 5 --engine [python, c] --async --operations greyscale, blur, sobel, gamma, histogram --blur-size 5 --blur-sigma 1.0 --gamma-value 1.5
metetl analyze --csv data/MetObjects.csv --output-dir data/plots
metetl --help
```