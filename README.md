# Sentiment-Based Product Recommendation System

This project builds an end-to-end **Sentiment-Based Product Recommendation System** for an e-commerce platform called **Ebuss**. The system recommends products to users based on:

1. **Collaborative Filtering (Item-Based)** – to find products similar to what the user already likes
2. **Sentiment Analysis of Reviews** – to prioritise products with more positive customer feedback

The final solution is deployed as a **Flask web application** where a user can enter a username and receive the **Top-5 recommended products**.

---

## Features

✔ Cleans & preprocesses product review text  
✔ Extracts TF-IDF features  
✔ Trains multiple ML models for sentiment classification  
✔ Selects **Logistic Regression** as the final model  
✔ Builds **Item-Based Collaborative Filtering recommender**  
✔ Ranks products using predicted sentiment  
✔ Deploys as a Flask web app  
✔ Accepts username input & displays final Top-5 products

---

## Machine Learning Models Used

### **Sentiment Model**

`Logistic Regression (TF-IDF Features)`

Selected because it achieved the best balance between:

- accuracy
- recall for negative reviews
- ROC-AUC

---

### **Recommendation System**

`Item-Based Collaborative Filtering`

Uses cosine similarity between product vectors to recommend similar products.

---

## Project Structure

project/
│
├── app.py
├── model.py
├── recommender.pkl
├── reviews_processed.pkl
├── sentiment_label_encoder.pkl
├── sentiment_logreg_model.pkl
├── tfidf_vectorizer.pkl
├── data/
│ └── sample30.csv
│ └── DataAttributeDescription.csv
├── templates/
│ └── index.html
├── Sentiment_Based_Product_Recommendation_System.ipynb
├── requirements.txt
└── Procfile

---

## Environment & Library Versions

The project was developed and tested using the following library versions:

pandas: 2.3.3
numpy: 2.2.6
scikit-learn: 1.7.2
xgboost: 3.1.2
nltk: 3.9.2
flask: 3.1.2
wordcloud: 1.9.5

## Requirements

To run the project, install the required Python libraries using:

- pip install -r requirements.txt

## Running Locally

```bash
pip install -r requirements.txt
python app.py

## Deployment

The project is deployed using:

- **Flask** — Backend web framework
- **Gunicorn** — Production web server
- **Heroku** — Cloud hosting (PaaS)

### Procfile

web: gunicorn app:app
```

## Dataset

- **Dataset Name:** `sample30.csv`
- **Size:** ~30,000 product reviews
- **Contents:** product details, ratings, review text, and sentiment labels

This dataset is used to train the sentiment model and build the recommendation engine.

---

## Final Output

The web app returns:
Top 5 Recommended Products

These products are ranked in **descending order of predicted positive sentiment score**, ensuring that users receive both relevant **and well-reviewed** item recommendations.

---

## Author

**Vaishali Makwana**  
Machine Learning Engineer — _Ebuss (Case Study Project)_

---

## Objective

To improve product recommendations by incorporating **customer sentiment insights**, ensuring that **highly-rated and positively-reviewed products are prioritised** for each user.

---
