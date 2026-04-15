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
from typing import Generator, Tuple

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd


FILE_PATH = 'MetObjects.csv'
CHUNK_SIZE = 10000
DATE_RANGE = (-5000, 2026)
CATEGORICAL_COLUMNS = ['Culture', 'Medium', 'Classification', 'Country']


def read_chunks(filepath: str,
                chunksize: int = CHUNK_SIZE) -> Generator[pd.DataFrame, None, None]:
    """
    Генератор, читающий файл частями.

    Args:
        filepath (str): Путь к CSV-файлу.
        chunksize (int): Размер чанка.

    Yields:
        pd.DataFrame: Очередной чанк данных.
    """
    reader = pd.read_csv(filepath, chunksize=chunksize, encoding='utf-8-sig')
    for chunk in reader:
        yield chunk


def prepare_chunk(
    chunk_iter: Generator[pd.DataFrame, None, None],
) -> Generator[pd.DataFrame, None, None]:
    """
    Генератор, приводящий данные к нужному для обработки виду.

    - Отбирает нужные столбцы.
    - Преобразует Object Begin Date в число.
    - Фильтрует по разумному диапазону.
    - Удаляет строки с NaN в дате.

    Args:
        chunk_iter: Итератор чанков.

    Yields:
        pd.DataFrame: Очищенный чанк.
    """
    for chunk in chunk_iter:
        df = chunk[['Department', 'Object Begin Date'] + CATEGORICAL_COLUMNS].copy()
        # Преобразуем дату
        df['Object Begin Date'] = pd.to_numeric(df['Object Begin Date'], errors='coerce')
        # Фильтруем даты
        mask = (df['Object Begin Date'] >= DATE_RANGE[0]) & (df['Object Begin Date'] <= DATE_RANGE[1])  # noqa E501
        df = df[mask]
        df = df.dropna(subset=['Object Begin Date'])
        yield df


def aggregate_chunk(
    chunk_iter: Generator[pd.DataFrame, None, None],
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]:
    """
    Генератор, обрабатывающий один чанк и возвращающий три агрегата:
    - статистика по отделам,
    - счётчик (отдел, год),
    - частоты категориальных значений.

    Args:
        chunk_iter: Генератор по чанкам

    Yields:
        Кортеж трёх промежуточных датафреймов
    """
    for df in chunk_iter:
        dept_agg = df.groupby("Department")["Object Begin Date"].agg(
            count="size",
            sum="sum",
            sum2=lambda x: (x**2).sum(),
        )

        year_agg = df.groupby(["Department", "Object Begin Date"]).size().rename("count")

        year_agg_df = year_agg.to_frame() if not year_agg.empty else pd.DataFrame(columns=["count"])  # noqa E501

        cat_agg = pd.DataFrame(index=pd.Index([], dtype=object),
                               columns=CATEGORICAL_COLUMNS)
        for col in CATEGORICAL_COLUMNS:
            values = df[col].dropna().astype(str).str.strip()
            values = values[values != ""]
            if not values.empty:
                freq = values.value_counts().rename(col).to_frame()
                cat_agg = cat_agg.add(freq, fill_value=0)

        yield dept_agg, year_agg_df, cat_agg


def collect_aggregates(
    agg_iter: Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Принимает итератор агрегатов чанков и собирает итоговые таблицы.

    Args:
        agg_iter: Генератор, возвращающий три датафрейма на каждой итерации

    Returns:
        Три сагрегированных из генератора датафрейма
    """
    dept_stats = pd.DataFrame(columns=["count", "sum", "sum2"]).astype(
        {"count": "int64", "sum": "float64", "sum2": "float64"},
    )

    empty_multi_index = pd.MultiIndex(
        levels=[[], []], codes=[[], []],
        names=["Department", "Object Begin Date"],
    )
    year_counts = pd.DataFrame(index=empty_multi_index,
                               columns=["count"]).astype({"count": "int64"})

    cat_df = pd.DataFrame(
        index=pd.Index([], dtype=object),
        columns=CATEGORICAL_COLUMNS,
    )

    for dept_agg, year_agg, cat_agg in agg_iter:
        dept_stats = dept_stats.add(dept_agg, fill_value=0)

        if not year_agg.empty:
            year_counts = year_counts.add(year_agg, fill_value=0)

        if not cat_agg.empty:
            cat_df = cat_df.add(cat_agg, fill_value=0)

    if not cat_df.empty:
        cat_df = cat_df.fillna(0).astype("int64")

    return dept_stats, year_counts, cat_df


def compute_department_stats(dept_stats: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    По накопленным агрегатам dept_stats (count, sum, sum2) вычисляет:
    mean, std, 95% CI, 95% scatter.

    Args:
        dept_stats (DataFrame): Накопленные агрегаты

    Returns:
        df_stats (DataFrame): Метрики и имя отдела с максимальным std.
        max_std_dept (str): Департамент с максимальным std.
    """
    # dept_stats имеет индекс Department и столбцы count, sum, sum2
    results = []
    for dept, row in dept_stats.iterrows():
        cnt = row['count']
        mean = row['sum'] / cnt
        variance = (row['sum2'] / cnt) - mean**2
        std = np.sqrt(variance)
        ci_margin = 1.96 * std / np.sqrt(cnt)
        scatter_margin = 1.96 * std
        results.append({
            'Department': dept,
            'count': cnt,
            'mean': mean,
            'std': std,
            'ci_lower': mean - ci_margin,
            'ci_upper': mean + ci_margin,
            'scatter_lower': mean - scatter_margin,
            'scatter_upper': mean + scatter_margin,
        })

    df_stats = pd.DataFrame(results)
    if df_stats.empty:
        return df_stats, ''

    max_std_dept = df_stats.loc[df_stats['std'].idxmax(), 'Department']

    return df_stats, max_std_dept


def compute_categorical_metrics(cat_df: pd.DataFrame) -> pd.DataFrame:
    """
    По DataFrame cat_df (столбцы — категориальные поля, индекс — значения категорий)
    вычисляет Gini, нормализованную энтропию, ENC.

    Args:
        cat_df: Стандартные метрики.

    Returns:
        metrics_list (DataFrame): Итоговые метрики.
    """
    metrics_list = []
    for col in cat_df.columns:
        counts = cat_df[col]
        n = counts.sum()
        if n == 0:
            continue

        probs = counts / n
        gini = 1 - (probs**2).sum()
        entropy = -(probs * np.log(probs)).sum()
        n_categories = (counts > 0).sum()
        norm_entropy = entropy / np.log(n_categories) if n_categories > 1 else 0
        enc = np.exp(entropy)
        metrics_list.append({
            'Column': col,
            'Gini': gini,
            'NormEntropy': norm_entropy,
            'ENC': enc,
            'Categories': n_categories,
            'Total': n,
        })

    return pd.DataFrame(metrics_list)


def plot_department_ages(df_stats: pd.DataFrame) -> None:
    """
    Построение столбцовой диаграммы.

    Args:
        df_stats (DataFrame): Полученный пайплайном результат для визуализации.
    """
    depts = df_stats['Department']
    means = df_stats['mean']
    ci_err = df_stats['ci_upper'] - means
    scat_err = df_stats['scatter_upper'] - means

    plt.figure(figsize=(12, 6))
    x = np.arange(len(depts))
    plt.bar(x, means, yerr=ci_err, capsize=5, label='95% CI', alpha=0.7)
    plt.errorbar(x, means, yerr=scat_err, fmt='o', color='red', label='95% scatter')
    plt.xticks(x, depts, rotation=45, ha='right')
    plt.ylabel('Average Object Begin Date (year)')
    plt.title('Average Age by Department')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Lab3\\department_age.png')
    plt.show()


def plot_timeline(year_counts: pd.DataFrame, department: str) -> None:
    """
    Временной график для заданного отдела.

    Args:
        year_counts: DataFrame с мультииндексом (Department, Year) и столбцом 'count'.
        department (str): Департамент.
    """
    dept_year_counts = year_counts.xs(department, level='Department')
    dept_year_counts = dept_year_counts.sort_index()
    window = 10
    rolling_avg = dept_year_counts['count'].rolling(window=window, center=True).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(dept_year_counts.index, dept_year_counts['count'], alpha=0.5, label='Count per year')  # noqa E501
    plt.plot(rolling_avg.index, rolling_avg.values, color='red',
             label=f'Moving avg (window={window})')
    plt.xlabel('Year')
    plt.ylabel('Number of objects created')
    plt.title(f'Creation timeline for {department}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Lab3\\timeline.png')
    plt.show()


def plot_heatmap(metrics_df: pd.DataFrame) -> None:
    """
    Строит тепловую карту метрик (нормированных по столбцам).

    Args:
        metrics_df (DataFrame): Словарь метрик для каждого поля.
    """
    cols = metrics_df['Column'].values
    metric_names = ['Gini', 'NormEntropy', 'ENC']
    data = metrics_df[metric_names].values

    # Нормализация по столбцам
    data_norm = data.copy()
    for i in range(data.shape[1]):
        col_vals = data[:, i]
        if col_vals.max() - col_vals.min() > 0:
            data_norm[:, i] = col_vals / col_vals.max()
        else:
            data_norm[:, i] = 0.0

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
    print(metrics_df.to_string(index=False))


def main() -> None:
    """Запуск анализа."""
    chunks = read_chunks(FILE_PATH, CHUNK_SIZE)
    prepared = prepare_chunk(chunks)
    aggregated_chunk = aggregate_chunk(prepared)
    dept_stats, year_counts, cat_df = collect_aggregates(aggregated_chunk)

    df_stats, max_std_dept = compute_department_stats(dept_stats)
    print(f"Отдел с максимальным разбросом (std): {max_std_dept}")
    plot_department_ages(df_stats)
    plot_timeline(year_counts, max_std_dept)

    metrics_df = compute_categorical_metrics(cat_df)
    plot_heatmap(metrics_df)


if __name__ == '__main__':
    main()
