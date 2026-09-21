from pathlib import Path
import sys

# Добавляем корневую папку в путь Python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data import загрузить_данные_смс
from src.preprocessing import предобработать_в_текст

КОРЕНЬ = Path(__file__).resolve().parents[1]
ОТЧЕТЫ = КОРЕНЬ / 'reports'
ОТЧЕТЫ.mkdir(exist_ok=True)

df = загрузить_данные_смс()
df['обработанный_текст'] = df.сообщение.map(предобработать_в_текст)

df[['метка','сообщение','обработанный_текст']].head(100).to_csv(ОТЧЕТЫ / 'неделя2_до_после.csv', index=False)

df['кол_слов_исходный'] = df.сообщение.str.split().str.len()
df['кол_слов_обработанный'] = df.обработанный_текст.str.split().str.len()
df[['кол_слов_исходный','кол_слов_обработанный']].describe().to_csv(ОТЧЕТЫ / 'неделя2_сравнение_длины.csv')

print(df[['сообщение','обработанный_текст']].head(10).to_string(index=False))
