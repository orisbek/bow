from collections import Counter
from typing import Iterable, List

from .preprocessing import предобработать


def _получить_лемматизатор():
    """Инициализирует лемматизатор pymorphy2 для русского языка."""
    try:
        import pymorphy2
        return pymorphy2.MorphAnalyzer()
    except ImportError:
        return None


def основные_формы(тексты: Iterable[str]) -> List[List[str]]:
    """Получает основные формы слов (леммы) для русских текстов.
    
    Args:
        тексты: Итерируемое значение текстов
        
    Returns:
        Список списков лемм
    """
    морф = _получить_лемматизатор()
    if морф is None:
        return None, 'pymorphy2 не установлен'
    
    результаты = []
    for текст in тексты:
        леммы = []
        токены = предобработать(текст, удалять_стоп_слова=False)
        for токена in токены:
            разбор = морф.parse(токена)[0]
            леммы.append(разбор.normal_form)
        результаты.append(леммы)
    
    return результаты, None


def топ_токены(df, n=50):
    """Получает топ N токенов из датасета.
    
    Args:
        df: DataFrame с колонкой 'сообщение'
        n: Количество топ токенов
        
    Returns:
        Список кортежей (токен, частота)
    """
    счетчик = Counter(
        токен for текст in df['сообщение'] 
        for токен in предобработать(текст)
    )
    return счетчик.most_common(n)
