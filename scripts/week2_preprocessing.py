from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data import load_sms_data
from src.preprocessing import preprocess_to_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'; OUT.mkdir(exist_ok=True)
df = load_sms_data()
df['processed_text'] = df.message.map(preprocess_to_text)
df[['label','message','processed_text']].head(100).to_csv(OUT / 'week2_before_after.csv', index=False)
df['raw_word_count'] = df.message.str.split().str.len()
df['processed_word_count'] = df.processed_text.str.split().str.len()
df[['raw_word_count','processed_word_count']].describe().to_csv(OUT / 'week2_length_comparison.csv')
print(df[['message','processed_text']].head(10).to_string(index=False))
