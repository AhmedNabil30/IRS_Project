# recommender.py
import pandas as pd
import numpy as np
import re
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

#########################################################
# 1) Load and preprocess dataset
#########################################################
file_path = r"C:\Users\medoh\OneDrive\Desktop\AIE425 Intelligent recommender systems\Project\FINALDATA.csv"
data = pd.read_csv(file_path)

# data = data.drop(
#     columns=["is_free","supported_languages","header_image","website","categories"],
#     errors="ignore"
# )

data = data.fillna("")

# Clean short_description
data["short_description"] = data["short_description"].str.lower()
data["short_description"] = data["short_description"].apply(
    lambda x: re.sub(r"[^a-z0-9\s]", "", x).strip()
)

# Ensure these exist
genre_cols = ["genre1","genre2","genre3","genre4","genre5","genres6","genres7"]
for gc in genre_cols:
    if gc not in data.columns:
        data[gc] = ""

# Clean name
data["clean_name"] = data["name"].str.lower()
data["clean_name"] = data["clean_name"].apply(
    lambda x: re.sub(r"[^a-z0-9\s]", "", x).strip()
)

#########################################################
# 2) Cluster game names automatically to unify versions
#########################################################

# Collect unique names
unique_names = data["clean_name"].unique()

# TF-IDF on names
tfidf_vec = TfidfVectorizer(stop_words="english").fit(unique_names)
name_tfidf = tfidf_vec.transform(unique_names)

# DBSCAN with cosine distance
dbscan_model = DBSCAN(
    eps=0.3,  # adjust as needed
    min_samples=1,
    metric="cosine"
)
cluster_labels = dbscan_model.fit_predict(name_tfidf)

# Build cluster mapping: { cluster_label: [list_of_names] }
cluster_map = {}
for label, n in zip(cluster_labels, unique_names):
    cluster_map.setdefault(label, []).append(n)

# Shortest name as representative
cluster_rep = {}
for lbl, name_list in cluster_map.items():
    rep = min(name_list, key=len)
    cluster_rep[lbl] = rep

# Reverse dict: any name -> representative
name_to_rep = {}
for lbl, name_list in cluster_map.items():
    rep = cluster_rep[lbl]
    for nm in name_list:
        name_to_rep[nm] = rep

data["unified_name"] = data["clean_name"].apply(lambda x: name_to_rep[x])

#########################################################
# 3) Build grouped rows: unify short descriptions, etc.
#########################################################
# grouped_rows = []
# grouped_unames = data["unified_name"].unique()

# for uname in grouped_unames:
#     subset = data[data["unified_name"]==uname]
#     merged_desc = " ".join(subset["short_description"].tolist())
#     avg_rating = subset["ratings"].astype(float).mean()
#     max_age = subset["required_age"].astype(float).max()
#     all_genres = []
#     for _, row in subset.iterrows():
#         for gc in genre_cols:
#             val = row[gc]
#             if val != "":
#                 all_genres.append(val)
#     all_genres = list(set(all_genres))
#     grouped_rows.append({
#         "unified_name": uname,
#         "required_age": max_age,
#         "ratings": avg_rating,
#         "merged_description": merged_desc,
#         "merged_genres": all_genres
#     })

# df_grouped = pd.DataFrame(grouped_rows)
# ...
grouped_rows = []
grouped_unames = data["unified_name"].unique()

for uname in grouped_unames:
    subset = data[data["unified_name"] == uname]
    
    merged_desc = " ".join(subset["short_description"].tolist())
    avg_rating = subset["ratings"].astype(float).mean()
    max_age = subset["required_age"].astype(float).max()
    
    # Merge genres
    all_genres = []
    for _, row in subset.iterrows():
        for gc in genre_cols:
            val = row[gc]
            if val != "":
                all_genres.append(val)
    all_genres = list(set(all_genres))
    
    # -------------- NEW: Merge additional fields --------------
    # 1) is_free: True if at least one row is free
    merged_is_free = subset["is_free"].astype(bool).any()
    
    # 2) supported_languages: combine all unique
    langs_set = set()
    for lang_val in subset["supported_languages"]:
        # Each row might have a string of languages
        if isinstance(lang_val, str):
            # Example splitting by comma or semicolon
            parts = [x.strip() for x in lang_val.split(",")]
            for p in parts:
                if p:
                    langs_set.add(p)
    merged_supported_languages = ", ".join(sorted(langs_set))
    
    # 3) header_image: pick the first non-empty or just the first in the subset
    #    (use subset.iloc[0] if you want the first row, or something else)
    merged_header = ""
    for val in subset["header_image"]:
        if val.strip():
            merged_header = val
            break
    
    # 4) website: pick the first or combine
    merged_website = ""
    for val in subset["website"]:
        if val.strip():
            merged_website = val
            break
    
    # 5) categories: combine similarly
    cat_set = set()
    if "categories" in subset.columns:
        for cat_val in subset["categories"]:
            if isinstance(cat_val, str):
                cat_parts = [x.strip() for x in cat_val.split(",")]
                for cp in cat_parts:
                    if cp:
                        cat_set.add(cp)
    merged_categories = ", ".join(sorted(cat_set))
    # ---------------------------------------------------------
    
    grouped_rows.append({
        "unified_name": uname,
        "required_age": max_age,
        "ratings": avg_rating,
        "merged_description": merged_desc,
        "merged_genres": all_genres,
        
        # NEW columns
        "is_free": merged_is_free,
        "supported_languages": merged_supported_languages,
        "header_image": merged_header,
        "website": merged_website,
        "categories": merged_categories
    })

df_grouped = pd.DataFrame(grouped_rows)


#########################################################
# 4) One-hot encode genres
#########################################################
all_g_set = set()
for g_list in df_grouped["merged_genres"]:
    for g in g_list:
        all_g_set.add(g)
all_genres_sorted = sorted(list(all_g_set))

def create_genre_vector(row_gs, full_gs):
    vec = [0]*len(full_gs)
    for g in row_gs:
        if g in full_gs:
            idx = full_gs.index(g)
            vec[idx] = 1
    return vec

genre_vectors = []
for gs in df_grouped["merged_genres"]:
    genre_vectors.append(create_genre_vector(gs, all_genres_sorted))
genre_vectors = np.array(genre_vectors, dtype=float)

#########################################################
# 5) TF-IDF on merged_description
#########################################################
desc_corpus = df_grouped["merged_description"].values
raw_vocab = set()
for doc in desc_corpus:
    words = doc.split()
    for w in words:
        raw_vocab.add(w)
raw_vocab_list = sorted(list(raw_vocab))

custom_sw = {
    "game","games","play","played","playing","world","players","new",
    "open","will","make","makes","made","thing","things","like"
}
filtered_vocab_list = [v for v in raw_vocab_list if v not in custom_sw]

def term_frequency(doc, vocab):
    counts = [0]*len(vocab)
    words = doc.split()
    for w in words:
        if w in vocab:
            idx = vocab.index(w)
            counts[idx] += 1
    return counts

tf_matrix = []
for doc in desc_corpus:
    tf_matrix.append(term_frequency(doc, filtered_vocab_list))
tf_matrix = np.array(tf_matrix, dtype=float)

doc_freq = np.count_nonzero(tf_matrix, axis=0)
N_docs = len(desc_corpus)

final_vocab = []
valid_idx = []
for i, w in enumerate(filtered_vocab_list):
    ratio = doc_freq[i]/N_docs
    if ratio <= 0.6:
        final_vocab.append(w)
        valid_idx.append(i)

tf_matrix_filtered = tf_matrix[:, valid_idx]
df_filtered = doc_freq[valid_idx]
idf_vals = []
for df_val in df_filtered:
    idf_vals.append(math.log((N_docs+1)/(df_val+1)) + 1)
idf_vals = np.array(idf_vals)

tf_idf_matrix = tf_matrix_filtered * idf_vals

def rowwise_norm(mat):
    out = []
    for row in mat:
        length = np.sqrt(np.sum(row**2))
        if length!=0:
            out.append(row/length)
        else:
            out.append(row)
    return np.array(out)

tf_idf_matrix = rowwise_norm(tf_idf_matrix)

#########################################################
# 6) Combine features (genres + TF-IDF), exclude ratings
#########################################################
combined_features = np.hstack([genre_vectors, tf_idf_matrix])

#########################################################
# 7) Manual SVD
#########################################################
def svd_decomposition(matrix, k=None):
    A = np.array(matrix, dtype=float)
    ATA = np.dot(A.T, A)
    AAT = np.dot(A, A.T)
    
    eigvals_ATA, V = np.linalg.eigh(ATA)
    eigvals_AAT, U = np.linalg.eigh(AAT)
    
    idx_v = np.argsort(eigvals_ATA)[::-1]
    idx_u = np.argsort(eigvals_AAT)[::-1]
    
    V = V[:, idx_v]
    U = U[:, idx_u]
    eigvals = eigvals_ATA[idx_v]
    
    if k is not None:
        U = U[:, :k]
        V = V[:, :k]
        eigvals = eigvals[:k]
    
    sigma = np.sqrt(eigvals)
    sigma_matrix = np.diag(sigma)
    return U, sigma_matrix, V.T

K = 30
U, S, Vt = svd_decomposition(combined_features, k=K)
reconstructed = np.dot(np.dot(U, S), Vt)

#########################################################
# 8) Build item-item similarity
#########################################################
def cosine_similarity_matrix(features):
    norms = np.sqrt(np.sum(features**2, axis=1, keepdims=True))
    normed = features/(norms+1e-8)
    sim = np.dot(normed, normed.T)
    return sim

item_sim_matrix = cosine_similarity_matrix(reconstructed)

#########################################################
# 9) Recommendation function
#########################################################
def recommend_games(
    user_age,    # "below_15" or "above_15"
    user_games,  # list of original names
    user_genres, # list of favored genres
    top_n=5
):
    if user_age=="below_15":
        valid_df = df_grouped[df_grouped["required_age"]<=15].copy()
    else:
        valid_df = df_grouped.copy()
    
    valid_idx = valid_df.index.tolist()
    
    def clean(gm):
        return re.sub(r"[^a-z0-9\s]", "", gm.lower()).strip()
    
    user_cleaned = [clean(gm) for gm in user_games]
    
    rep_indices = []
    for uc in user_cleaned:
        if uc in name_to_rep:
            cluster_rep = name_to_rep[uc]
            row_match = valid_df[valid_df["unified_name"]==cluster_rep]
            if len(row_match)>0:
                ridx = row_match.index[0]
                rep_indices.append(ridx)
        else:
            row_match = valid_df[valid_df["unified_name"]==uc]
            if len(row_match)>0:
                ridx = row_match.index[0]
                rep_indices.append(ridx)
    
    if len(rep_indices)==0:
        valid_df["similarity_score"] = valid_df["ratings"].astype(float)
        valid_df.sort_values("similarity_score", ascending=False, inplace=True)
        return valid_df.head(top_n)
    
    avg_scores = []
    for idx in valid_idx:
        local_sims = []
        for r_idx in rep_indices:
            local_sims.append(item_sim_matrix[idx, r_idx])
        avg_scores.append(np.mean(local_sims))
    valid_df["similarity_score"] = avg_scores
    
    def measure_overlap(row_gs, user_gs):
        row_set = set(row_gs)
        user_set = set(user_gs)
        overlap_count = len(row_set.intersection(user_set))
        if len(user_gs)==0:
            return 0
        return overlap_count/len(user_gs)
    
    overlap_scores = []
    for i, row in valid_df.iterrows():
        overlap_scores.append(measure_overlap(row["merged_genres"], user_genres))
    valid_df["genre_overlap"] = overlap_scores
    
    w_sim = 0.2
    w_genre = 0.5
    w_rat = 0.3
    final_score = []
    for i, row in valid_df.iterrows():
        score = (
            w_sim * row["similarity_score"] +
            w_genre * row["genre_overlap"] +
            w_rat * row["ratings"]
        )
        final_score.append(score)
    valid_df["combined_score"] = final_score
    
    valid_df.sort_values("combined_score", ascending=False, inplace=True)
    top_recs = valid_df.head(top_n).copy()
    
    sim_details = []
    for i, row in top_recs.iterrows():
        txt = ""
        for r_idx in rep_indices:
            rep_name = df_grouped.loc[r_idx,"unified_name"]
            val_sim = item_sim_matrix[i, r_idx]
            txt += f"  Similar to {rep_name} => {val_sim:.3f}\n"
        sim_details.append(txt.strip())
    top_recs["similarity_details"] = sim_details
    
    return top_recs
