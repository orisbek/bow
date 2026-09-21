from pathlib import Path
import sys

# Добавляем корневую папку в путь Python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from src.data import загрузить_данные_смс

КОРЕНЬ = Path(__file__).resolve().parents[1]
ОТЧЕТЫ = КОРЕНЬ / 'reports'
ОТЧЕТЫ.mkdir(exist_ok=True)

df = загрузить_данные_смс()
df['кол_символов'] = df.сообщение.str.len()
df['кол_слов'] = df.сообщение.str.split().str.len()

статистика = pd.DataFrame({
    'строк': [len(df)],
    'пропущенные_значения': [int(df.isna().sum().sum())],
    'дубликаты_по_сообщению': [int(df.duplicated('сообщение').sum())],
})

статистика.to_csv(ОТЧЕТЫ / 'неделя1_статистика.csv', index=False)
df.groupby('метка')[['кол_символов','кол_слов']].describe().to_csv(ОТЧЕТЫ / 'неделя1_сгруппированная_статистика.csv')

ax = df.метка.value_counts().plot(kind='bar', title='Распределение классов SMS')
ax.figure.tight_layout()
ax.figure.savefig(ОТЧЕТЫ / 'неделя1_распределение_классов.png', dpi=160)
plt.close(ax.figure)

print(статистика.to_string(index=False))
print(df.метка.value_counts())
