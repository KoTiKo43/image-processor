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
from collections import Counter, defaultdict
from typing import Any, Dict, Generator, Iterable, List, Tuple

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd


FILE_PATH = 'MetObjects.csv'
CHUNK_SIZE = 10000
CATEGORICAL_COLUMNS = ['Department', 'Culture', 'Medium', 'Classification', 'Country']


def read_chunks(filepath: str,
                chunksize: int = CHUNK_SIZE) -> Generator[pd.DataFrame, None, None]:
    """
    Генератор, читающий файл частями.

    Args:
        filepath: путь к CSV-файлу.
        chunksize: размер чанка.

    Yields:
        pd.DataFrame: очередной чанк данных.
    """
    reader = pd.read_csv(filepath, chunksize=chunksize, encoding='utf-8-sig')
    for chunk in reader:
        yield chunk


def process_department_data(
    chunk_iter: Generator[pd.DataFrame, None, None]
) -> Generator[Dict[str, Dict[str, Any]], None, None]:
    """
    Генератор, накапливающий для каждого отдела статистику и список дат.

    Args:
        chunk_iter: генератор чанков.

    Yields:
        Словарь: ключ – отдел, значение – словарь с полями:
            count, sum, sum2, dates.
    """
    dept_data = defaultdict(lambda: {'count': 0, 'sum': 0.0, 'sum2': 0.0, 'dates': []})
    for chunk in chunk_iter:
        df = chunk[['Department', 'Object Begin Date']].copy()
        df['Object Begin Date'] = pd.to_numeric(df['Object Begin Date'], errors='coerce')
        # Расчёт интерквартильного размаха
        q1 = df['Object Begin Date'].quantile(0.25)
        q3 = df['Object Begin Date'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Отрезание того, что не попадает в интерквартильный размах
        df = df[(df['Object Begin Date'] >= lower_bound) & (df['Object Begin Date'] <= upper_bound)]  # noqa E501
        df = df.dropna(subset=['Object Begin Date'])
        for _, row in df.iterrows():
            dept = row['Department']
            date = row['Object Begin Date']
            dept_data[dept]['count'] += 1
            dept_data[dept]['sum'] += date
            dept_data[dept]['sum2'] += date * date
            dept_data[dept]['dates'].append(date)

    yield dept_data


def compute_stats(
    dept_data: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Вычисляет финальные метрики для каждого отдела.

    Args:
        dept_data: накопленная статистика (count, sum, sum2, dates).

    Returns:
        results: словарь с метриками (mean, std, ci_lower, ci_upper,
                 scatter_lower, scatter_upper, dates).
        max_std_dept: отдел с максимальным стандартным отклонением.
    """
    results = {}
    max_std = -1.0
    max_std_dept = ''
    for dept, data in dept_data.items():
        cnt = data['count']
        mean = data['sum'] / cnt
        variance = (data['sum2'] / cnt) - mean ** 2
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
            'dates': data['dates']
        }
        if std > max_std:
            max_std = std
            max_std_dept = dept

    return results, max_std_dept


def compute_categorical_metrics(filepath: str, columns: List[str],
                                chunksize: int = 10000) -> Dict[str, Dict[str, Any]]:
    """
    Вычисляет метрики для категориальных полей.

    Args:
        filepath (str): Путь к CSV-файлу.
        columns (List): Список имён столбцов.
        chunksize (int): Размер куска.

    Returns:
        Dict: для каждого поля gini, norm_entropy, enc, num_categories, total_entries.
    """
    counters = {col: Counter() for col in columns}
    reader = pd.read_csv(filepath, chunksize=chunksize, encoding='utf-8-sig',
                         usecols=columns)
    for chunk in reader:
        for col in columns:
            values = chunk[col].dropna().astype(str).str.strip()
            values = values[values != '']
            counters[col].update(values)

    metrics = {}
    for col, cnt in counters.items():
        n = sum(cnt.values())
        if n == 0:
            continue

        probs = np.array([c / n for c in cnt.values()])
        gini = 1 - np.sum(probs**2)
        entropy = -np.sum(probs * np.log(probs))
        n_counts = len(cnt)
        norm_entropy = entropy / np.log(n_counts) if n_counts > 1 else 0
        enc = np.exp(entropy)
        metrics[col] = {
            'gini': gini,
            'norm_entropy': norm_entropy,
            'enc': enc,
            'num_categories': n_counts,
            'total_entries': n
        }

    return metrics


def plot_department_ages(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Построение столбцовой диаграммы.

    Args:
        results (Dict): Полученный пайплайном результат для визуализации.
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


def plot_timeline(department: str, dates: List[float]) -> None:
    """
    Временной график количества созданных объектов со скользящим средним.

    Args:
        department: название отдела.
        dates: список годов создания.
    """
    year_counts = pd.Series(dates).value_counts().sort_index()
    window = 10
    rolling_avg = year_counts.rolling(window=window, center=True).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(year_counts.index, year_counts.values, alpha=0.5, label='Count per year')
    plt.plot(rolling_avg.index, rolling_avg.values, color='red',
             label=f'Moving avg (window={window})')
    plt.xlabel('Year')
    plt.ylabel('Number of objects created')
    plt.title(f'Creation timeline for {department}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Lab3\\timeline.png')
    plt.show()


def plot_heatmap(metrics: Dict[str, Dict[str, Any]]) -> None:
    """
    Строит тепловую карту метрик (нормированных по столбцам).

    Args:
        metrics (Dict): Словарь метрик для каждого поля.
    """
    cols = list(metrics.keys())
    metric_names = ['gini', 'norm_entropy', 'enc']
    data = np.array([[metrics[col][m] for m in metric_names] for col in cols])

    # Нормализация по столбцам
    data_norm = data.copy()
    for i in range(data.shape[1]):
        col_vals = data[:, i]
        max_val = col_vals.max()
        if max_val > 0:
            data_norm[:, i] = col_vals / max_val
        else:
            data_norm[:, i] = 0

    plt.figure(figsize=(8, len(cols) * 0.5 + 1))
    im = plt.imshow(data_norm, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    plt.xticks(np.arange(len(metric_names)), metric_names, rotation=45, ha='right')
    plt.yticks(np.arange(len(cols)), cols)
    plt.colorbar(im, label='Normalized value (higher = better quality)')
    plt.title('Categorical Data Quality Metrics')
    plt.tight_layout()
    plt.savefig('Lab3\\categorical_heatmap.png')
    plt.show()

    print("\nMetrics per column:")
    for col in cols:
        m = metrics[col]
        print(f"{col}: Gini={m['gini']:.4f}, NormEntropy={m['norm_entropy']:.4f}, "
              f"ENC={m['enc']:.2f} (categories={m['num_categories']})")


def main() -> None:
    """Запуск анализа."""
    chunks = read_chunks(FILE_PATH, CHUNK_SIZE)
    data_iter = process_department_data(chunks)
    dept_data = next(data_iter)
    results, max_std_dept = compute_stats(dept_data)
    print(f"Отдел с максимальным разбросом (std): {max_std_dept}")
    plot_department_ages(results)
    dates_for_timeline = results[max_std_dept]['dates']
    plot_timeline(max_std_dept, dates_for_timeline)

    metrics = compute_categorical_metrics(FILE_PATH, CATEGORICAL_COLUMNS, CHUNK_SIZE)
    plot_heatmap(metrics)


if __name__ == '__main__':
    main()
