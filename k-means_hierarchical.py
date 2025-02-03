import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import re

from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cdist

from gensim.models import Word2Vec
from torch.utils.data import DataLoader

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import warnings

warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')

DATA_PATH = '/kaggle/input/clickbait-dataset/clickbait_data.csv'
df = pd.read_csv(DATA_PATH)

plt.figure(figsize=(6, 4))
sns.countplot(x='clickbait', data=df, palette='viridis')
plt.title("Dataset Distribution: Clickbait vs. Non-Clickbait")
plt.xlabel("Clickbait Label")
plt.ylabel("Count")
plt.xticks([0, 1], ["Non-Clickbait (0)", "Clickbait (1)"])
plt.show()


# TEXT PREPROCESSING
def preprocess_text(text):
    """
    Cleans the input text by removing punctuation and numbers using regex,
    converting to lowercase, tokenizing, and eliminating English stopwords.
    """
    # Remove punctuation
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    # Remove digits
    text_no_digits = re.sub(r'\d+', '', text_no_punct)
    # Convert text to lowercase
    text_lower = text_no_digits.lower()
    # Tokenize
    tokens = word_tokenize(text_lower)
    # Remove stopwords
    stopword_set = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stopword_set]
    # Join back
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text


df['cleaned_headline'] = df['headline'].apply(preprocess_text)

#     TRAIN/TEST SPLIT
X = df['cleaned_headline']
y = df['clickbait']

train_texts, test_texts, train_labels, test_labels = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#   FEATURE EXTRACTION

# ---- 1) TF-IDF ----
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_train_features = tfidf_vectorizer.fit_transform(train_texts)
tfidf_test_features = tfidf_vectorizer.transform(test_texts)

print(f"TF-IDF Train Shape: {tfidf_train_features.shape}")
print(f"TF-IDF Test Shape : {tfidf_test_features.shape}")

# ---- 2) Word2Vec ----
tokenized_train_texts = [t.split() for t in train_texts]
tokenized_test_texts = [t.split() for t in test_texts]

word2vec_model = Word2Vec(
    sentences=tokenized_train_texts,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    seed=42
)


def average_word2vec_embedding(tokens, model, embedding_dim=100):
    valid_vectors = [model.wv[token] for token in tokens if token in model.wv]
    if valid_vectors:
        return np.mean(valid_vectors, axis=0)
    else:
        return np.zeros(embedding_dim)


word2vec_train = np.array([
    average_word2vec_embedding(tokens, word2vec_model) for tokens in tokenized_train_texts
])
word2vec_test = np.array([
    average_word2vec_embedding(tokens, word2vec_model) for tokens in tokenized_test_texts
])

# ---- 3) BERT ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.to(device)

from torch.utils.data import DataLoader


def embed_texts_in_batches(text_list, model, tokenizer, batch_size=32, max_length=512):
    dataloader = DataLoader(text_list, batch_size=batch_size)
    all_embeddings = []
    for batch in dataloader:
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        # pooler_output shape: (batch_size, hidden_size)
        embeddings = outputs.pooler_output.cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)


print("Generating BERT embeddings for train/test...")
bert_train = embed_texts_in_batches(train_texts.tolist(), bert_model, bert_tokenizer, batch_size=32)
bert_test = embed_texts_in_batches(test_texts.tolist(), bert_model, bert_tokenizer, batch_size=32)

# Feature Scaling
scaler = StandardScaler()

word2vec_train_scaled = scaler.fit_transform(word2vec_train)
word2vec_test_scaled = scaler.transform(word2vec_test)

bert_train_scaled = scaler.fit_transform(bert_train)
bert_test_scaled = scaler.transform(bert_test)


# K-MEANS CLUSTERING
def kmeans_hparam_search(
        train_data,
        ground_truth,
        feature_label='Features',
        inits_to_try=['k-means++', 'random'],
        n_inits_to_try=[5, 10],
        max_iters_to_try=[100, 300]
):
    """
    Manually searches hyperparams (init, n_init, max_iter) for KMeans (with n_clusters=2).
    Prints each iteration's results and returns best config based on ARI on the train set.
    """
    results = []
    for init_method in inits_to_try:
        for n_init_value in n_inits_to_try:
            for max_iter_value in max_iters_to_try:
                kmeans = KMeans(
                    n_clusters=2,
                    init=init_method,
                    n_init=n_init_value,
                    max_iter=max_iter_value,
                    random_state=42
                )
                kmeans.fit(train_data)
                labels_pred = kmeans.labels_
                ari_on_train = adjusted_rand_score(ground_truth, labels_pred)

                # Print the result for iteration
                print(f"{feature_label} | init={init_method}, n_init={n_init_value}, "
                      f"max_iter={max_iter_value}, ARI={ari_on_train:.4f}")

                results.append({
                    'init': init_method,
                    'n_init': n_init_value,
                    'max_iter': max_iter_value,
                    'ARI': ari_on_train
                })

    best_config = max(results, key=lambda x: x['ARI'])
    print(f"\n=== [K-Means] {feature_label} ===")
    print(f"Best Configuration by Train ARI: {best_config}")
    return best_config, results


# A helper to visualize final clusters via PCA
def visualize_clusters(feature_data, cluster_labels, title, n_components=2):
    """
    Reduces feature data to n_components using PCA and plots the clusters.
    """
    pca = PCA(n_components=n_components, random_state=42)
    reduced_data = pca.fit_transform(feature_data)
    df_pca = pd.DataFrame({
        'PC1': reduced_data[:, 0],
        'PC2': reduced_data[:, 1],
        'Cluster': cluster_labels
    })
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df_pca,
        x='PC1', y='PC2',
        hue='Cluster',
        palette='viridis',
        alpha=0.7
    )
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()


# === K-Means on TF-IDF
best_cfg_tfidf, _ = kmeans_hparam_search(
    tfidf_train_features, train_labels, feature_label='TF-IDF'
)
kmeans_tfidf = KMeans(
    n_clusters=2,
    init=best_cfg_tfidf['init'],
    n_init=best_cfg_tfidf['n_init'],
    max_iter=best_cfg_tfidf['max_iter'],
    random_state=42
)
kmeans_tfidf.fit(tfidf_train_features)
tfidf_train_clusters = kmeans_tfidf.labels_
tfidf_test_clusters = kmeans_tfidf.predict(tfidf_test_features)
tfidf_test_ari = adjusted_rand_score(test_labels, tfidf_test_clusters)
print(f"\n[K-Means: TF-IDF] Test ARI: {tfidf_test_ari:.4f}")
visualize_clusters(
    tfidf_train_features.toarray(),
    tfidf_train_clusters,
    title='K-Means on TF-IDF (Train)'
)

# === K-Means on Word2Vec
best_cfg_w2v, _ = kmeans_hparam_search(
    word2vec_train_scaled, train_labels, feature_label='Word2Vec'
)
kmeans_w2v = KMeans(
    n_clusters=2,
    init=best_cfg_w2v['init'],
    n_init=best_cfg_w2v['n_init'],
    max_iter=best_cfg_w2v['max_iter'],
    random_state=42
)
kmeans_w2v.fit(word2vec_train_scaled)
w2v_train_clusters = kmeans_w2v.labels_
w2v_test_clusters = kmeans_w2v.predict(word2vec_test_scaled)
w2v_test_ari = adjusted_rand_score(test_labels, w2v_test_clusters)
print(f"\n[K-Means: Word2Vec] Test ARI: {w2v_test_ari:.4f}")
visualize_clusters(
    word2vec_train_scaled,
    w2v_train_clusters,
    title='K-Means on Word2Vec (Train)'
)

# === K-Means on BERT
best_cfg_bert, _ = kmeans_hparam_search(
    bert_train_scaled, train_labels, feature_label='BERT'
)
kmeans_bert = KMeans(
    n_clusters=2,
    init=best_cfg_bert['init'],
    n_init=best_cfg_bert['n_init'],
    max_iter=best_cfg_bert['max_iter'],
    random_state=42
)
kmeans_bert.fit(bert_train_scaled)
bert_train_clusters = kmeans_bert.labels_
bert_test_clusters = kmeans_bert.predict(bert_test_scaled)
bert_test_ari = adjusted_rand_score(test_labels, bert_test_clusters)
print(f"\n[K-Means: BERT] Test ARI: {bert_test_ari:.4f}")
visualize_clusters(
    bert_train_scaled,
    bert_train_clusters,
    title='K-Means on BERT (Train)'
)



# HIERARCHICAL CLUSTERING

def hierarchical_hparam_search(
        train_data,
        ground_truth,
        feature_label='Features',
        linkages=['ward', 'complete', 'average'],
        affinities=['euclidean', 'manhattan', 'cosine']
):
    """
    Searches over specified linkages and affinities for AgglomerativeClustering 
    with n_clusters=2. Prints each iteration's ARI. Returns best config.
    """
    results = []

    if hasattr(train_data, 'toarray'):
        train_data = train_data.toarray()

    for linkage in linkages:
        for affinity in affinities:
            # 'ward' only works with 'euclidean'
            if linkage == 'ward' and affinity != 'euclidean':
                continue

            clusterer = AgglomerativeClustering(
                n_clusters=2,
                linkage=linkage,
                affinity=affinity
            )
            labels_pred = clusterer.fit_predict(train_data)
            ari_score = adjusted_rand_score(ground_truth, labels_pred)

            # Print iteration result
            print(f"{feature_label} | linkage={linkage}, affinity={affinity}, ARI={ari_score:.4f}")

            results.append({
                'linkage': linkage,
                'affinity': affinity,
                'ARI': ari_score
            })

    best_config = max(results, key=lambda x: x['ARI'])
    print(f"\n=== [Hierarchical] {feature_label} ===")
    print(f"Best Configuration by Train ARI: {best_config}")
    return best_config, results


def compute_cluster_centers(X_train, train_labels, n_clusters=2):
    centers = []
    for cluster_id in range(n_clusters):
        cluster_points = X_train[train_labels == cluster_id]
        center = np.mean(cluster_points, axis=0)
        centers.append(center)
    return np.vstack(centers)


def assign_nearest_center(X_test, centers):
    distances = cdist(X_test, centers, metric='euclidean')
    predicted_labels = np.argmin(distances, axis=1)
    return predicted_labels


# --- Hierarchical on TF-IDF ---
best_cfg_tfidf_hc, _ = hierarchical_hparam_search(
    tfidf_train_features,
    train_labels,
    feature_label='TF-IDF (HC)'
)
# Re-fit with best config on train
tfidf_train_dense = tfidf_train_features.toarray()
best_linkage = best_cfg_tfidf_hc['linkage']
best_affinity = best_cfg_tfidf_hc['affinity']
hc_tfidf = AgglomerativeClustering(
    n_clusters=2,
    linkage=best_linkage,
    affinity=best_affinity
)
tfidf_train_clusters_hc = hc_tfidf.fit_predict(tfidf_train_dense)
# Compute cluster centers on train
tfidf_centers = compute_cluster_centers(tfidf_train_dense, tfidf_train_clusters_hc, n_clusters=2)
# Predict on test
tfidf_test_dense = tfidf_test_features.toarray()
tfidf_test_clusters_hc = assign_nearest_center(tfidf_test_dense, tfidf_centers)
tfidf_test_ari_hc = adjusted_rand_score(test_labels, tfidf_test_clusters_hc)
print(f"\n[Hierarchical: TF-IDF] Test ARI: {tfidf_test_ari_hc:.4f}")
visualize_clusters(
    tfidf_train_dense,
    tfidf_train_clusters_hc,
    title='Hierarchical on TF-IDF (Train)'
)

# --- Hierarchical on Word2Vec ---
best_cfg_w2v_hc, _ = hierarchical_hparam_search(
    word2vec_train_scaled,
    train_labels,
    feature_label='Word2Vec (HC)'
)
best_linkage = best_cfg_w2v_hc['linkage']
best_affinity = best_cfg_w2v_hc['affinity']
hc_w2v = AgglomerativeClustering(
    n_clusters=2,
    linkage=best_linkage,
    affinity=best_affinity
)
train_clusters_w2v_hc = hc_w2v.fit_predict(word2vec_train_scaled)
w2v_centers = compute_cluster_centers(word2vec_train_scaled, train_clusters_w2v_hc, n_clusters=2)
test_clusters_w2v_hc = assign_nearest_center(word2vec_test_scaled, w2v_centers)
w2v_test_ari_hc = adjusted_rand_score(test_labels, test_clusters_w2v_hc)
print(f"\n[Hierarchical: Word2Vec] Test ARI: {w2v_test_ari_hc:.4f}")
visualize_clusters(
    word2vec_train_scaled,
    train_clusters_w2v_hc,
    title='Hierarchical on Word2Vec (Train)'
)

# --- Hierarchical on BERT ---
best_cfg_bert_hc, _ = hierarchical_hparam_search(
    bert_train_scaled,
    train_labels,
    feature_label='BERT (HC)'
)
best_linkage = best_cfg_bert_hc['linkage']
best_affinity = best_cfg_bert_hc['affinity']
hc_bert = AgglomerativeClustering(
    n_clusters=2,
    linkage=best_linkage,
    affinity=best_affinity
)
train_clusters_bert_hc = hc_bert.fit_predict(bert_train_scaled)
bert_centers = compute_cluster_centers(bert_train_scaled, train_clusters_bert_hc, n_clusters=2)
test_clusters_bert_hc = assign_nearest_center(bert_test_scaled, bert_centers)
bert_test_ari_hc = adjusted_rand_score(test_labels, test_clusters_bert_hc)
print(f"\n[Hierarchical: BERT] Test ARI: {bert_test_ari_hc:.4f}")
visualize_clusters(
    bert_train_scaled,
    train_clusters_bert_hc,
    title='Hierarchical on BERT (Train)'
)

random_predictions = np.random.randint(0, 2, size=len(test_labels))
random_ari = adjusted_rand_score(test_labels, random_predictions)
print(f"\n[Random Chance] Test ARI: {random_ari:.4f}")

# SUPERVISED BASELINE
logreg_tfidf = LogisticRegression(random_state=42)
logreg_tfidf.fit(tfidf_train_features, train_labels)
pred_tfidf = logreg_tfidf.predict(tfidf_test_features)
acc_tfidf = accuracy_score(test_labels, pred_tfidf)
f1_tfidf = f1_score(test_labels, pred_tfidf)
ari_tfidf = adjusted_rand_score(test_labels, pred_tfidf)
print(f"\n[Supervised: TF-IDF] Accuracy: {acc_tfidf:.4f}, F1: {f1_tfidf:.4f}, ARI: {ari_tfidf:.4f}")

logreg_w2v = LogisticRegression(random_state=42)
logreg_w2v.fit(word2vec_train_scaled, train_labels)
pred_w2v = logreg_w2v.predict(word2vec_test_scaled)
acc_w2v = accuracy_score(test_labels, pred_w2v)
f1_w2v = f1_score(test_labels, pred_w2v)
ari_w2v = adjusted_rand_score(test_labels, pred_w2v)
print(f"[Supervised: Word2Vec] Accuracy: {acc_w2v:.4f}, F1: {f1_w2v:.4f}, ARI: {ari_w2v:.4f}")

logreg_bert = LogisticRegression(random_state=42)
logreg_bert.fit(bert_train_scaled, train_labels)
pred_bert = logreg_bert.predict(bert_test_scaled)
acc_bert = accuracy_score(test_labels, pred_bert)
f1_bert = f1_score(test_labels, pred_bert)
ari_bert = adjusted_rand_score(test_labels, pred_bert)
print(f"[Supervised: BERT] Accuracy: {acc_bert:.4f}, F1: {f1_bert:.4f}, ARI: {ari_bert:.4f}")