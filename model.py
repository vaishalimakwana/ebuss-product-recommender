import pickle
import numpy as np
import pandas as pd

# -------- Load Saved Models --------
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("sentiment_logreg_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

with open("recommender.pkl", "rb") as f:
    rec_data = pickle.load(f)

ui_full = rec_data["ui_full"]
item_sim_full = rec_data["item_sim_full"]


# -------- Item-based prediction --------
def predict_item_based_full(user, item):
    if user not in ui_full.index or item not in ui_full.columns:
        return np.nan
    
    user_ratings = ui_full.loc[user]
    sims = item_sim_full[item]
    mask = user_ratings > 0
    
    if mask.sum() == 0:
        return np.nan
    
    denom = sims[mask].sum()
    if denom == 0:
        return np.nan
    
    return np.dot(sims[mask], user_ratings[mask]) / denom


# -------- Top-N recommender --------
def recommend_top_n_items_full(user, n=20):
    if user not in ui_full.index:
        return []
    
    user_ratings = ui_full.loc[user]
    already = set(user_ratings[user_ratings > 0].index)
    
    preds = []
    for item in ui_full.columns:
        if item in already:
            continue
        
        score = predict_item_based_full(user, item)
        if not np.isnan(score):
            preds.append((item, score))
    
    preds.sort(key=lambda x: x[1], reverse=True)
    
    return [x[0] for x in preds[:n]]


# -------- Sentiment scoring --------
def sentiment_score_for_product(product_name, reviews_df):
    prod_reviews = reviews_df[reviews_df["name"] == product_name]
    
    if prod_reviews.shape[0] == 0:
        return None
    
    X = tfidf.transform(prod_reviews["processed_text"].tolist())
    preds = sentiment_model.predict(X)
    
    return (preds == 1).mean() * 100


# -------- FINAL FUNCTION CALLED BY FLASK --------
def recommend_top5_sentiment(user, reviews_df):
    top20 = recommend_top_n_items_full(user, n=20)

    scores = []
    for p in top20:
        score = sentiment_score_for_product(p, reviews_df)
        if score is not None:
            scores.append((p, score))

    if len(scores) == 0:
        return []

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:5]