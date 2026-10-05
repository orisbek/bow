from pathlib import Path
import pandas as pd
from collections import Counter
from src.data import load_sms_data
from src.preprocessing import preprocess
from src.linguistics import stems, spacy_lemmas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'; OUT.mkdir(exist_ok=True)
df = load_sms_data()
rows = []
for text in df.message.head(300):
    tokens = preprocess(text)
    porter, snowball = stems(tokens)
    rows.append({'text': text, 'tokens': ' '.join(tokens), 'porter': ' '.join(porter), 'snowball': ' '.join(snowball)})
pd.DataFrame(rows).to_csv(OUT / 'week3_stemming_examples.csv', index=False)
lemmas, error = spacy_lemmas(df.message.head(300))
if lemmas is not None:
    pd.DataFrame({'message': df.message.head(300), 'lemmas': [' '.join(x) for x in lemmas]}).to_csv(OUT / 'week3_spacy_lemmas.csv', index=False)
else:
    (OUT / 'week3_spacy_status.txt').write_text(error, encoding='utf-8')

tokens = [t for text in df.message for t in preprocess(text)]
porter, snowball = stems(tokens)
pd.DataFrame(Counter(tokens).most_common(50), columns=['term','count']).to_csv(OUT / 'week3_top50_tokens.csv', index=False)
pd.DataFrame(Counter(porter).most_common(50), columns=['term','count']).to_csv(OUT / 'week3_top50_porter.csv', index=False)
pd.DataFrame(Counter(snowball).most_common(50), columns=['term','count']).to_csv(OUT / 'week3_top50_snowball.csv', index=False)
print('Week 3 outputs written to reports/')
