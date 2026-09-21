from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_sms_data
from src.preprocessing import preprocess_to_text
from src.vectorization import fit_transform_train_test

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
IMAGES = ROOT / 'outputs' / 'images'
REPORTS.mkdir(exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)


def save_top_terms_plot(bow_terms: pd.DataFrame, tfidf_terms: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    bow_plot = bow_terms.sort_values('weight').tail(20)
    axes[0].barh(bow_plot['term'], bow_plot['weight'])
    axes[0].set_title('BOW: ╤é╨╛╨┐-20 ╨┐╨╛ ╤ü╤â╨╝╨╝╨░╤Ç╨╜╨╛╨╣ ╤ç╨░╤ü╤é╨╛╤é╨╡')
    axes[0].set_xlabel('╨Ü╨╛╨╗╨╕╤ç╨╡╤ü╤é╨▓╨╛ ╨▓╤à╨╛╨╢╨┤╨╡╨╜╨╕╨╣')
    axes[0].grid(axis='x', alpha=0.35)
    axes[0].set_axisbelow(True)

    tfidf_plot = tfidf_terms.sort_values('weight').tail(20)
    axes[1].barh(tfidf_plot['term'], tfidf_plot['weight'])
    axes[1].set_title('TF-IDF: ╤é╨╛╨┐-20 ╨┐╨╛ ╤ü╤Ç╨╡╨┤╨╜╨╡╨╝╤â ╨▓╨╡╤ü╤â')
    axes[1].set_xlabel('╨í╤Ç╨╡╨┤╨╜╨╕╨╣ TF-IDF')
    axes[1].grid(axis='x', alpha=0.35)
    axes[1].set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(IMAGES / 'week4_bow_tfidf_top20.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_wordclouds(bow_terms: pd.DataFrame, tfidf_terms: pd.DataFrame) -> None:
    try:
        from wordcloud import WordCloud
    except ImportError:
        (REPORTS / 'wordcloud_status.txt').write_text(
            '╨ƒ╨░╨║╨╡╤é wordcloud ╨╜╨╡ ╤â╤ü╤é╨░╨╜╨╛╨▓╨╗╨╡╨╜. ╨Æ╤ï╨┐╨╛╨╗╨╜╨╕╤é╨╡: pip install wordcloud',
            encoding='utf-8',
        )
        return

    for name, terms, filename, title in [
        ('BOW', bow_terms, 'week4_bow_wordcloud.png', '╨₧╨▒╨╗╨░╨║╨╛ ╤ü╨╗╨╛╨▓ BOW'),
        ('TF-IDF', tfidf_terms, 'week4_tfidf_wordcloud.png', '╨₧╨▒╨╗╨░╨║╨╛ ╤ü╨╗╨╛╨▓ TF-IDF'),
    ]:
        frequencies = dict(zip(terms['term'], terms['weight']))
        cloud = WordCloud(
            width=1500,
            height=650,
            background_color='white',
            colormap='viridis' if name == 'BOW' else 'autumn',
            max_words=100,
            collocations=False,
            random_state=42,
        ).generate_from_frequencies(frequencies)

        plt.figure(figsize=(16, 7))
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(IMAGES / filename, dpi=220, bbox_inches='tight')
        plt.close()


df = load_sms_data()
df['processed_text'] = df.message.map(preprocess_to_text)
train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df.label,
)
vecs = fit_transform_train_test(train.processed_text, test.processed_text)
rows = []
bow_terms = None
tfidf_terms = None

for name, (vectorizer, X_train, X_test) in vecs.items():
    rows.append({
        'representation': name,
        'train_rows': X_train.shape[0],
        'test_rows': X_test.shape[0],
        'features': X_train.shape[1],
        'train_nonzero': int(X_train.nnz),
        'test_nonzero': int(X_test.nnz),
    })
    terms = vectorizer.get_feature_names_out()
    if name == 'bow':
        weights = X_train.sum(axis=0).A1
    else:
        weights = X_train.mean(axis=0).A1

    top = (
        pd.DataFrame({'term': terms, 'weight': weights})
        .sort_values('weight', ascending=False)
        .head(100)
        .reset_index(drop=True)
    )
    top.to_csv(REPORTS / f'week4_top100_{name}.csv', index=False)

    if name == 'bow':
        bow_terms = top
    else:
        tfidf_terms = top

pd.DataFrame(rows).to_csv(REPORTS / 'week4_matrix_comparison.csv', index=False)

if bow_terms is not None and tfidf_terms is not None:
    save_top_terms_plot(bow_terms, tfidf_terms)
    save_wordclouds(bow_terms, tfidf_terms)

df[['label', 'message', 'processed_text']].to_csv(
    REPORTS / 'week4_processed_dataset.csv', index=False
)
train[['label', 'message', 'processed_text']].to_csv(
    REPORTS / 'week4_train.csv', index=False
)
test[['label', 'message', 'processed_text']].to_csv(
    REPORTS / 'week4_test.csv', index=False
)

print(pd.DataFrame(rows).to_string(index=False))
print(f'╨ÿ╨╖╨╛╨▒╤Ç╨░╨╢╨╡╨╜╨╕╤Å ╤ü╨╛╤à╤Ç╨░╨╜╨╡╨╜╤ï ╨▓: {IMAGES}')
