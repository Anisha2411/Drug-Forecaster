
from flask import Flask, request, render_template
import requests
import joblib
import numpy as np
import json
from bs4 import BeautifulSoup
import re
import time

app = Flask(__name__)

# Load ML model for drug rating prediction
model = joblib.load("rating_model.pkl")

# Map pregnancy categories to numerical values
preg_map = {'A':1, 'B':2, 'C':3, 'D':4, 'X':5}

def extract_side_effects(info):
    """Extract side effects from multiple FDA fields and format them"""
    # Combine all possible side effect sources
    sources = [info.get(field) for field in [
        "adverse_reactions", "warnings", "precautions", 
        "boxed_warning", "contraindications", "general_precautions"]]
    
    # Clean and split into readable chunks
    text = " ".join(str(s) for s in sources if s)
    return [text[i:i+200] for i in range(0, len(text), 200)] if text else ["No side effects data"]

def get_drug_alternatives(drug_name):
    """Find alternative drugs using RxNorm API"""
    try:
        # Get RxNorm ID for the drug
        rxcui = requests.get(
            f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}&search=1"
        ).json().get('idGroup',{}).get('rxnormId',[None])[0]
        
        if not rxcui: return ["No alternatives found"]
        
        # Get related drugs (both ingredients and brand names)
        concepts = requests.get(
            f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json?tty=IN+BN"
        ).json().get('relatedGroup',{}).get('conceptGroup',[])
        
        # Extract unique alternative names
        alts = {p['name'] for g in concepts if g.get('conceptProperties') 
               for p in g['conceptProperties'] if p.get('name') and p['name'].lower() != drug_name.lower()}
        
        return list(alts)[:5] or ["No alternatives found"]
        
    except Exception as e:
        print(f"Alternative drugs error: {e}")
        return ["Alternative search failed"]

def predict_rating(side_effects, preg_cat):
    """Predict drug rating using trained model"""
    try:
        rating = model.predict([[len(side_effects), preg_map.get(preg_cat,3)]])[0]
        return {
            'Predicted Rating': round(rating,1),
            'Normal User Risk': 'Low' if rating>=8 else 'Moderate' if rating>=5 else 'High',
            'Pregnant Woman Risk': {
                'A':'Safe', 'B':'Likely Safe', 'C':'Caution',
                'D':'Unsafe', 'X':'Contraindicated'}.get(preg_cat,'Unknown'),
            'Pregnancy Category': preg_cat
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

import requests
from bs4 import BeautifulSoup

def get_drug_image(drug_name):
    try:
        url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=pageimages&piprop=original&titles={drug_name}"
        resp = requests.get(url, timeout=10).json()
        pages = resp["query"]["pages"]
        for _, p in pages.items():
            if "original" in p:
                return p["original"]["source"]
        return None
    except Exception as e:
        print("Wikimedia fetch error:", e)
        return None

def scrape_apollo_price(drug_name):
    """Scrape price from Apollo Pharmacy"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        search_url = f"https://www.apollopharmacy.in/search-medicines/{drug_name.replace(' ', '%20')}"
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for price elements (adjust selectors based on actual site structure)
            price_elements = soup.find_all(['span', 'div'], class_=re.compile(r'price|cost|amount', re.I))
            
            for element in price_elements:
                text = element.get_text().strip()
                price_match = re.search(r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
                if price_match:
                    return f"₹{price_match.group(1)}"
        
        return "50-100 Estimate (Check website)"
        
    except Exception as e:
        print(f"Apollo scraping error: {e}")
        return "50-100 Estimate (Check website)"

def scrape_pharmeasy_price(drug_name):
    """Scrape price from PharmEasy"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        search_url = f"https://pharmeasy.in/search/all?name={drug_name.replace(' ', '%20')}"
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for price elements
            price_elements = soup.find_all(['span', 'div', 'p'], class_=re.compile(r'price|cost|amount|mrp', re.I))
            
            for element in price_elements:
                text = element.get_text().strip()
                price_match = re.search(r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
                if price_match:
                    return f"₹{price_match.group(1)}"
        
        return "Price not available"
        
    except Exception as e:
        print(f"PharmEasy scraping error: {e}")
        return "Price not available"

def scrape_1mg_price(drug_name):
    """Scrape price from 1mg"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        search_url = f"https://www.1mg.com/search/all?name={drug_name.replace(' ', '%20')}"
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for price elements
            price_elements = soup.find_all(['span', 'div'], class_=re.compile(r'price|cost|amount', re.I))
            
            for element in price_elements:
                text = element.get_text().strip()
                price_match = re.search(r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
                if price_match:
                    return f"₹{price_match.group(1)}"
        
        return "Price not available"
        
    except Exception as e:
        print(f"1mg scraping error: {e}")
        return "Price not available"

def get_drug_prices(drug_name):
    """Get real-time drug prices from Indian pharmacy websites"""
    try:
        print(f"Fetching prices for: {drug_name}")
        
        # Add small delays between requests to be respectful
        apollo_price = scrape_apollo_price(drug_name)
        time.sleep(1)
        
        pharmeasy_price = scrape_pharmeasy_price(drug_name)
        time.sleep(1)
        
        onemg_price = scrape_1mg_price(drug_name)
        
        pharmacies = [
            {
                'name': 'Apollo Pharmacy',
                'price': apollo_price,
                'url': f'https://www.apollopharmacy.in/search-medicines/{drug_name.replace(" ", "%20")}',
                'availability': 'Check Website'
            },
            {
                'name': 'PharmEasy',
                'price': pharmeasy_price,
                'url': f'https://pharmeasy.in/search/all?name={drug_name.replace(" ", "%20")}',
                'availability': 'Check Website'
            },
            {
                'name': '1mg',
                'price': onemg_price,
                'url': f'https://www.1mg.com/search/all?name={drug_name.replace(" ", "%20")}',
                'availability': 'Check Website'
            }
        ]
        
        return pharmacies
        
    except Exception as e:
        print(f"Price fetch error: {e}")
        return []

def get_drug_details(drug_name):
    """Main function to get all drug data from FDA API"""
    try:
        # Get drug data from FDA
        data = requests.get(
            f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{drug_name}&limit=1",
            timeout=10
        ).json().get("results",[None])[0]
        
        if not data: return {"error": "Drug not found in FDA database"}
        
        # Process all data
        side_effects = extract_side_effects(data)
        preg_cat = data.get("pregnancy_category",["C"])[0] if isinstance(
            data.get("pregnancy_category"),list) else data.get("pregnancy_category","C")
        
        return {
            "prediction": predict_rating(side_effects, preg_cat),
            "side_effects": side_effects,
            "warnings": data.get("warnings",["No warnings"]),
            "usage": data.get("indications_and_usage",["No usage info"]),
            "brand_names": list(set(data.get("openfda",{}).get("brand_name",[drug_name]))),
            "alternative": get_drug_alternatives(drug_name),
            "image_url": get_drug_image(drug_name),
            "prices": get_drug_prices(drug_name)
        }
        
    except requests.exceptions.RequestException as e:
        return {"error": f"FDA API error: {str(e)}"}
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}

@app.route("/", methods=["GET","POST"])
def index():
    """Main Flask route handling requests"""
    if request.method == "POST":
        data = get_drug_details(request.form["drug_name"])
        if "error" in data: 
            return render_template("index.html", error=data["error"])
        return render_template("index.html",
            prediction=data["prediction"],
            extra_info={
                "Side Effects": data["side_effects"],
                "Usage": data["usage"][0] if isinstance(data["usage"],list) else data["usage"],
                "Warnings": data["warnings"][0] if isinstance(data["warnings"],list) else data["warnings"],
                "Brand Names": ", ".join(data["brand_names"]),
                "Alternative Drugs": data["alternative"],
                "Pregnancy Category": data["prediction"]["Pregnancy Category"]
            },
            image_url=data.get("image_url"),
            prices=data.get("prices", []))
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

