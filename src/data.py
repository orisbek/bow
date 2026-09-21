from pathlib import Path
import pandas as pd


def корневая_папка() -> Path:
    """Возвращает корневую папку проекта."""
    return Path(__file__).resolve().parents[1]


def загрузить_данные_смс(путь=None) -> pd.DataFrame:
    """Загружает данные SMS из файла.
    
    Args:
        путь: Путь к файлу данных
        
    Returns:
        DataFrame с колонками 'метка' и 'сообщение'
    """
    путь_данных = Path(путь) if путь else корневая_папка() / 'data' / 'SMSSpamCollection'
    df = pd.read_csv(путь_данных, sep='\t', header=None, names=['метка', 'сообщение'], encoding='utf-8')
    df = df.dropna(subset=['метка', 'сообщение']).copy()
    df['метка'] = df['метка'].str.strip().str.lower()
    df['сообщение'] = df['сообщение'].astype(str)
    return df
