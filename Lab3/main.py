"""
Исследование и оценка информации о датасете с помощью пайплайна.
Вариант 1
Анализ возраста объектов по отделам
1. Для каждого из отделов музея (Department) найти средний возраст
объектов (Object Begin Date), 95% доверительный интервал и 95%
интервал рассеяния. Построить столбцовую диаграмму.
2. Для отдела музея с максимальным разбросом дат создания объектов
построить временной график создания объектов, со скользящим
средним.
"""
from collections import defaultdict
from typing import Dict, Iterable

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd


def run_pipeline(filepath: str, chunksize: int = 10000) -> Dict:
    """
    Пайплайн, обрабатывающий массив данных.

    Args:
        filepath (str): Путь до массива данных (csv).
        chunksize (int): Размер блока.

    Returns:
        result (dict): Словарь с результатами.
    """
    reader = pd.read_csv(filepath, chunksize=chunksize, encoding='utf-8-sig')

    def extract(iterator: Iterable) -> :
        """
        Итератор, извлекающий Department и Object Begin Date.
        Приводит дату к числу.
        Удаляет None значения.

        Args:
            iterator: Итератор

        Yields:
            df: Обработанная часть
        """
        for chunk in iterator:
            df = chunk[['Department', 'Object Begin Date']].copy()
            df['Object Begin Date'] = pd.to_numeric(df['Object Begin Date'],
                                                    errors='coerce')
            df = df.dropna(subset='Object Begin Date')
            q1 = df['Object Begin Date'].quantile(0.25)
            q3 = df['Object Begin Date'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df = df[(df['Object Begin Date'] >= lower_bound) & (df['Object Begin Date'] <= upper_bound)]
            yield df

    data_iter = extract(reader)

    stats = defaultdict(lambda: {'count': 0, 'sum': 0, 'sum2': 0})
    for chunk in data_iter:
        for _, row in chunk.iterrows():
            dept = row['Department']
            date = row['Object Begin Date']
            stats[dept]['count'] += 1
            stats[dept]['sum'] += date
            stats[dept]['sum2'] += date * date

    results = {}
    for dept, s in stats.items():
        cnt = s['count']
        if cnt < 2:
            continue  # недостаточно данных для статистики

        mean = s['sum'] / cnt
        variance = (s['sum2'] / cnt) - mean**2
        std = np.sqrt(variance)
        ci_margin = 1.96 * std / np.sqrt(cnt)
        scatter_margin = 1.96 * std
        results[dept] = {
            'count': cnt,
            'mean': mean,
            'std': std,
            'ci_lower': mean - ci_margin,
            'ci_upper': mean + ci_margin,
            'scatter_lower': mean - scatter_margin,
            'scatter_upper': mean + scatter_margin,
        }

    return results


def plot_department_ages(results: dict) -> None:
    """
    Построение столбцовой диаграммы.

    Args:
        results (dict): Полученный пайплайном результат для визуализации.
    """
    depts = list(results.keys())
    means = [results[d]['mean'] for d in depts]
    ci_err = [results[d]['ci_upper'] - results[d]['mean'] for d in depts]
    scat_err = [results[d]['scatter_upper'] - results[d]['mean'] for d in depts]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(depts))
    plt.bar(x, means, yerr=ci_err, capsize=5, label='95% CI', alpha=0.7)
    plt.errorbar(x, means, yerr=scat_err, fmt='o', color='red', label='95% scatter')
    plt.xticks(x, depts, rotation=45, ha='right')
    plt.ylabel('Average Object Begin Date')
    plt.title('Average Age by Department')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Lab3\\department_age.png')
    plt.show()


def timeline_for_max_spread(filepath: str, results: dict, chunksize: int = 10000) -> None:
    """
    Временной график для отдела с максимальным разбросом.

    Args:
        filepath (str): Путь до массива данных (csv).
        results (dict): Полученный пайплайном результат для визуализации.
        chunksize (int): Размер блока.
    """
    # Находим отдел с максимальным стандартным отклонением
    max_dept = max(results, key=lambda d: results[d]['std'])
    print(f"Department with highest spread: {max_dept}")

    # Второй проход: собираем все даты для этого отдела
    dates = []
    reader = pd.read_csv(filepath, chunksize=chunksize, encoding='utf-8-sig')
    for chunk in reader:
        mask = chunk['Department'] == max_dept
        chunk = chunk.loc[mask, ['Object Begin Date']].copy()
        chunk['Object Begin Date'] = pd.to_numeric(chunk['Object Begin Date'],
                                                   errors='coerce')
        chunk = chunk.dropna()
        dates.extend(chunk['Object Begin Date'].tolist())

    # Группировка по годам
    year_counts = pd.Series(dates).value_counts().sort_index()

    # Скользящее среднее (окно 10 лет)
    window = 10
    rolling_avg = year_counts.rolling(window=window, center=True).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(year_counts.index, year_counts.values, alpha=0.5, label='Count per year')
    plt.plot(rolling_avg.index, rolling_avg.values, color='red',
             label=f'Moving average (window={window})')
    plt.xlabel('Year')
    plt.ylabel('Number of objects created')
    plt.title(f'Creation timeline for {max_dept}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Lab3\\timeline.png')
    plt.show()


if __name__ == '__main__':
    filepath = 'MetObjects.csv'
    results = run_pipeline(filepath, chunksize=10000)
    plot_department_ages(results)
    timeline_for_max_spread(filepath, results, chunksize=10000)
