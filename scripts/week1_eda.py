from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from src.data import load_sms_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)
df = load_sms_data()
df['char_count'] = df.message.str.len()
df['word_count'] = df.message.str.split().str.len()
summary = pd.DataFrame({
    'rows': [len(df)],
    'missing_values': [int(df.isna().sum().sum())],
    'duplicates_by_message': [int(df.duplicated('message').sum())],
})
summary.to_csv(OUT / 'week1_summary.csv', index=False)
df.groupby('label')[['char_count','word_count']].describe().to_csv(OUT / 'week1_grouped_stats.csv')
ax = df.label.value_counts().plot(kind='bar', title='SMS class distribution')
ax.figure.tight_layout(); ax.figure.savefig(OUT / 'week1_class_distribution.png', dpi=160); plt.close(ax.figure)
print(summary.to_string(index=False))
print(df.label.value_counts())
