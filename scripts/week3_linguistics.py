from pathlib import Path
import sys

# Добавляем корневую папку в путь Python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from collections import Counter
from src.data import загрузить_данные_смс
from src.preprocessing import предобработать
from src.linguistics import основные_формы

КОРЕНЬ = Path(__file__).resolve().parents[1]
ОТЧЕТЫ = КОРЕНЬ / 'reports'
ОТЧЕТЫ.mkdir(exist_ok=True)

df = загрузить_данные_смс()
строки = []

# Примеры лемматизации
леммы, ошибка = основные_формы(df.сообщение.head(300))
if леммы is not None:
    pd.DataFrame({
        'сообщение': df.сообщение.head(300),
        'леммы': [' '.join(x) for x in леммы]
    }).to_csv(ОТЧЕТЫ / 'неделя3_леммы.csv', index=False)
else:
    (ОТЧЕТЫ / 'неделя3_статус.txt').write_text(ошибка, encoding='utf-8')

# Анализ частотности токенов
токены = [т for текст in df.сообщение for т in предобработать(текст)]
pd.DataFrame(Counter(токены).most_common(50), columns=['термин','кол_во']).to_csv(ОТЧЕТЫ / 'неделя3_топ50_токены.csv', index=False)

print('Отчеты недели 3 записаны в папку reports/')
