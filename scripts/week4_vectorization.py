from pathlib import Path
import sys

# Добавляем корневую папку в путь Python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import загрузить_данные_смс
from src.preprocessing import предобработать_в_текст
from src.vectorization import преобразовать_обучение_тест

КОРЕНЬ = Path(__file__).resolve().parents[1]
ОТЧЕТЫ = КОРЕНЬ / 'reports'
ИЗОБРАЖЕНИЯ = КОРЕНЬ / 'outputs' / 'images'
ОТЧЕТЫ.mkdir(exist_ok=True)
ИЗОБРАЖЕНИЯ.mkdir(parents=True, exist_ok=True)


def сохранить_график_топ_терминов(мешок_слов_термины: pd.DataFrame, тф_идф_термины: pd.DataFrame) -> None:
    """Сохраняет график топ 20 терминов для BOW и TF-IDF."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    граф_мешок = мешок_слов_термины.sort_values('вес').tail(20)
    axes[0].barh(граф_мешок['термин'], граф_мешок['вес'])
    axes[0].set_title('Мешок слов: топ-20 по суммарной частоте')
    axes[0].set_xlabel('Количество вхождений')
    axes[0].grid(axis='x', alpha=0.35)
    axes[0].set_axisbelow(True)

    граф_тф_идф = тф_идф_термины.sort_values('вес').tail(20)
    axes[1].barh(граф_тф_идф['термин'], граф_тф_идф['вес'])
    axes[1].set_title('TF-IDF: топ-20 по среднему весу')
    axes[1].set_xlabel('Средний TF-IDF')
    axes[1].grid(axis='x', alpha=0.35)
    axes[1].set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(ИЗОБРАЖЕНИЯ / 'неделя4_мешок_слов_тф_идф_топ20.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def сохранить_облака_слов(мешок_слов_термины: pd.DataFrame, тф_идф_термины: pd.DataFrame) -> None:
    """Сохраняет облака слов для BOW и TF-IDF."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        (ОТЧЕТЫ / 'облако_слов_статус.txt').write_text(
            'Пакет wordcloud не установлен. Выполните: pip install wordcloud',
            encoding='utf-8',
        )
        return

    for название, термины, имя_файла, заголовок in [
        ('Мешок слов', мешок_слов_термины, 'неделя4_мешок_слов_облако.png', 'Облако слов (Мешок слов)'),
        ('TF-IDF', тф_идф_термины, 'неделя4_тф_идф_облако.png', 'Облако слов (TF-IDF)'),
    ]:
        частоты = dict(zip(термины['термин'], термины['вес']))
        облако = WordCloud(
            width=1500,
            height=650,
            background_color='white',
            colormap='viridis' if название == 'Мешок слов' else 'autumn',
            max_words=100,
            collocations=False,
            random_state=42,
        ).generate_from_frequencies(частоты)

        plt.figure(figsize=(16, 7))
        plt.imshow(облако, interpolation='bilinear')
        plt.axis('off')
        plt.title(заголовок)
        plt.tight_layout(pad=0)
        plt.savefig(ИЗОБРАЖЕНИЯ / имя_файла, dpi=220, bbox_inches='tight')
        plt.close()


# Загружаем и обрабатываем данные
df = загрузить_данные_смс()
df['обработанный_текст'] = df.сообщение.map(предобработать_в_текст)

# Разделяем на тренировочную и тестовую выборки
обучение, тест = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df.метка,
)

# Преобразуем в векторы
векторы = преобразовать_обучение_тест(обучение.обработанный_текст, тест.обработанный_текст)
строки = []
мешок_слов_термины = None
тф_идф_термины = None

# Анализируем каждое представление
for название, (векторизатор, X_обучение, X_тест) in векторы.items():
    строки.append({
        'представление': название,
        'строк_обучение': X_обучение.shape[0],
        'строк_тест': X_тест.shape[0],
        'признаки': X_обучение.shape[1],
        'ненулевые_обучение': int(X_обучение.nnz),
        'ненулевые_тест': int(X_тест.nnz),
    })
    
    термины = векторизатор.get_feature_names_out()
    if название == 'мешок_слов':
        веса = X_обучение.sum(axis=0).A1
    else:
        веса = X_обучение.mean(axis=0).A1

    топ = (
        pd.DataFrame({'термин': термины, 'вес': веса})
        .sort_values('вес', ascending=False)
        .head(100)
        .reset_index(drop=True)
    )
    топ.to_csv(ОТЧЕТЫ / f'неделя4_топ100_{название}.csv', index=False)

    if название == 'мешок_слов':
        мешок_слов_термины = топ
    else:
        тф_идф_термины = топ

# Сохраняем результаты сравнения матриц
pd.DataFrame(строки).to_csv(ОТЧЕТЫ / 'неделя4_сравнение_матриц.csv', index=False)

# Сохраняем графики и облака слов
if мешок_слов_термины is not None and тф_идф_термины is not None:
    сохранить_график_топ_терминов(мешок_слов_термины, тф_идф_термины)
    сохранить_облака_слов(мешок_слов_термины, тф_идф_термины)

# Сохраняем обработанный датасет
df[['метка', 'сообщение', 'обработанный_текст']].to_csv(
    ОТЧЕТЫ / 'неделя4_обработанный_датасет.csv', index=False
)
обучение[['метка', 'сообщение', 'обработанный_текст']].to_csv(
    ОТЧЕТЫ / 'неделя4_обучение.csv', index=False
)
тест[['метка', 'сообщение', 'обработанный_текст']].to_csv(
    ОТЧЕТЫ / 'неделя4_тест.csv', index=False
)

# Выводим результаты
print(pd.DataFrame(строки).to_string(index=False))
print(f'Изображения сохранены в: {ИЗОБРАЖЕНИЯ}')
