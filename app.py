from flask import Flask, render_template, request
import pickle
from model import recommend_top5_sentiment

app = Flask(__name__)

# Load processed reviews data
with open("reviews_processed.pkl", "rb") as f:
    reviews_df = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    username = None
    error = None

    if request.method == "POST":
        username = request.form.get("username").strip().lower()

        try:
            recommendations = recommend_top5_sentiment(username, reviews_df)
            if len(recommendations) == 0:
                error = "No recommendations available for this user."
        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        recommendations=recommendations,
        username=username,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)