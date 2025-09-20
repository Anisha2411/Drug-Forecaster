from flask import Flask, request, render_template
import requests
import joblib
import numpy as np
from bs4 import BeautifulSoup
import os  # Add this import
from sklearn.linear_model import LinearRegression  # Add this import

app = Flask(__name__)

# Check if model file exists, if not create it
if not os.path.exists("rating_model.pkl"):
    print("Creating a new model file...")
    
    # Create and train a simple model
    model = LinearRegression()
    # Sample data: [pregnancy_category, side_effect_count]
    X = np.array([[1, 3], [2, 5], [3, 8], [4, 10], [5, 12]])
    # Corresponding ratings (1-10 scale)
    y = np.array([8, 7, 5, 3, 1])
    model.fit(X, y)
    
    # Save the model to a file
    joblib.dump(model, "rating_model.pkl")
    print("Model created and saved as rating_model.pkl")
else:
    # Load the existing model
    model = joblib.load("rating_model.pkl")

# Map pregnancy categories to numerical values
preg_map = {'A':1, 'B':2, 'C':3, 'D':4, 'X':5}

def extract_side_effects(info):
    """Extract side effects (dummy function for now)"""
    return info.get("side_effects", ["No side effects listed"])

def get_drug_price_info(drug_name):
    """Fetch dummy drug price info (you can replace with API later)"""
    prices = {
        "Paracetamol": "₹25 for 10 tablets",
        "Ibuprofen": "₹45 for 10 tablets",
        "Amoxicillin": "₹120 for 6 capsules"
    }
    return prices.get(drug_name, "Price info not available")

def get_drug_image(drug_name):
    """Fetch the first drug image from Google Images"""
    url = f"https://www.google.com/search?tbm=isch&q={drug_name}+medicine&safe=active"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        img_tag = soup.find("img")
        if img_tag and "src" in img_tag.attrs:
            return img_tag["src"]
    return None  # fallback if no image found

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        drug_name = request.form["drug_name"]
        preg_cat = request.form["preg_cat"]

        # Prepare model input
        preg_val = preg_map.get(preg_cat, 0)
        X = np.array([[preg_val]])
        prediction = model.predict(X)[0]

        # Collect extra info
        price_info = get_drug_price_info(drug_name)
        drug_image = get_drug_image(drug_name)

        return render_template(
            "result.html",
            drug_name=drug_name,
            prediction=prediction,
            price_info=price_info,
            drug_image=drug_image
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)