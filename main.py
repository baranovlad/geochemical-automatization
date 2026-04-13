import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def get_stats(log_data):
    """
    Расчет статистик для логарифмированного набора данных.
    :param log_data: Датафрейм с логарифмированными данными.
    :return:
    """
    stats = log_data.describe().T
    stats['mode'] = log_data.mode().iloc[0]
    stats['skew'] = log_data.skew()

    stats = stats.apply(np.exp)

    stats['cf.var(%)'] = stats['std'] / stats['mean'] * 100
    stats['first'] = stats['mean'] * stats['std']
    stats['second'] = stats['first'] * stats['std']
    stats['third'] = stats['second'] * stats['std']

    return stats


def get_kmeans_clusters(raw_data, general_stats, count_clusters=11, count_init=300):
    """
    Кластеризация данных методом KMeans.
    :param raw_data: Исходные данные
    :param general_stats: Генеральные характеристики данных
    :param count_clusters: Количество кластеров
    :param count_init: Количество инициализаий
    :return:
    """
    data = raw_data.drop(['Hg', 'Li', 'Cr', 'x msk', 'y msk'], axis=1).astype(float)

    # Применение аномальных уровней к данным
    for i, v in enumerate(data.columns):
        data.loc[(data[v] > general_stats.iloc[i]['second']), v] = general_stats.iloc[i]['second']

    # Нормализация данных
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_features = scaler.transform(data)
    scaled_data = pd.DataFrame(scaled_features, columns=data.columns)

    # Кластеризация данных
    k_means = KMeans(random_state=10, n_init=count_init, n_clusters=count_clusters)
    k_means.fit(scaled_data)
    labels = k_means.labels_
    return labels


def get_pivot_table(raw_data):
    """
    Создает сводную таблицу для кластеров.
    :param raw_data: Данные с проведенной кластеризацией
    :return: Сводная таблица
    """
    pivot = raw_data.drop(['x msk', 'y msk'], axis=1).pivot_table(index='cluster', aggfunc='mean')
    pivot['count'] = raw_data['cluster'].value_counts()
    return pivot


def plot_clusters(raw_data, output_path='map.html'):
    """
    Создает интерактивную карту с точками кластерной выборки.
    :param raw_data: Данные для вывода
    :param output_path: Путь для сохранения карты
    :return: None
    """
    markers = ['circle', 'square', 'triangle-up', 'diamond', 'cross', 'x', 'star', 'hexagon', 'pentagon']
    cluster_markers = {cluster: markers[i % len(markers)] for i, cluster in enumerate(raw_data['cluster'].unique())}

    fig = px.scatter(
        raw_data,
        x='x msk',
        y='y msk',
        color='cluster',
        symbol='cluster',
        title='Распределение точек кластерной выборки',
        labels={'x msk': 'Координата X', 'y msk': 'Координата Y', 'cluster': 'Кластер'},
        color_discrete_sequence=px.colors.qualitative.Set1,
        symbol_map=cluster_markers
    )
    fig.update_traces(marker=dict(size=10))
    fig.write_html(output_path)



def plot_heatmaps(raw_data):
    """
    Создает тепловые карты для каждого кластера.
    :param raw_data: Данные с метками кластеров.
    :return: None
    """
    clusters = raw_data['cluster'].unique()
    for cluster in clusters:
        cluster_data = raw_data[raw_data['cluster'] == cluster].drop(['x msk', 'y msk', 'cluster'], axis=1)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cluster_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title(f'Тепловая карта для кластера {cluster}')
        plt.savefig(f'heatmap_cluster_{cluster}.png', dpi=300)
        break


def save_to_excel(stats, clusts):
    """
    Сохраняет статистики и кластерные средние в Excel-файл.
    :param stats: Генерация стастистики
    :param clusts: Сводная таблица
    :return:
    """
    with pd.ExcelWriter(r'data.xlsx') as writer:
        stats.to_excel(writer, sheet_name='general_stats')
        clusts.T.to_excel(writer, sheet_name='cluster_means')


if __name__=='__main__':
    # Параметры
    path = 'bd_float.xlsx' # путь к файлу
    required_level = 'second' # номер аномальн
    count_clusters = 11
    count_init = 300

    # Загрузка данных
    raw_data = pd.read_excel(path, index_col=0)
    log_data = raw_data.drop(['x msk', 'y msk'], axis=1).apply(np.log)

    # Расчет статистик
    general_stats = get_stats(log_data)

    # Кластеризация
    clusters = get_kmeans_clusters(raw_data, general_stats, count_clusters, count_init)
    raw_data['cluster'] = clusters

    # Сводная таблица
    pivot_table = get_pivot_table(raw_data)

    # Сохранение результатов
    # plot_heatmaps(raw_data)
    # plot_clusters(raw_data)
    # save_to_excel(general_stats, pivot_table)

