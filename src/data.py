from pathlib import Path
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_sms_data(path=None) -> pd.DataFrame:
    data_path = Path(path) if path else project_root() / 'data' / 'SMSSpamCollection'
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'], encoding='utf-8')
    df = df.dropna(subset=['label', 'message']).copy()
    df['label'] = df['label'].str.strip().str.lower()
    df['message'] = df['message'].astype(str)
    return df
