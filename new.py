from flask import Flask, request, render_template
import requests
import joblib
import numpy as np
import json
from bs4 import BeautifulSoup
import re
import time
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

# Enhanced model loading with fallback
try:
    model = joblib.load("enhanced_rating_model.pkl")
    print("Loaded enhanced rating model")
except:
    try:
        model = joblib.load("rating_model.pkl")
        print("Loaded original rating model")
    except:
        print("No model found. Creating enhanced model...")
        model = create_enhanced_model()

# Map pregnancy categories to numerical values
preg_map = {'A':1, 'B':2, 'C':3, 'D':4, 'X':5}

def create_enhanced_model():
    """Create an enhanced model using comprehensive drug data"""
    try:
        # Try to load your existing dataset
        try:
            df = pd.read_csv("drug_dataset.csv")
            print(f"Loaded your dataset with {len(df)} records")
            
            # Use your existing features or create enhanced ones
            if 'side_effects_count' in df.columns and 'pregnancy_category' in df.columns:
                X = df[['side_effects_count', 'pregnancy_category']]
                y = df['rating']
            else:
                # Create enhanced training data
                training_data = create_comprehensive_training_data()
                df = pd.DataFrame(training_data)
                X = df[['side_effects_count', 'pregnancy_category_num']]
                y = df['rating']
                
        except FileNotFoundError:
            # Create comprehensive training data from common drugs
            training_data = create_comprehensive_training_data()
            df = pd.DataFrame(training_data)
            X = df[['side_effects_count', 'pregnancy_category_num']]
            y = df['rating']
        
        # Train enhanced model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        
        # Save the enhanced model
        joblib.dump(model, "enhanced_rating_model.pkl")
        print("Enhanced model created and saved")
        return model
        
    except Exception as e:
        print(f"Enhanced model creation failed: {e}")
        return create_fallback_model()

def create_comprehensive_training_data():
    """Create comprehensive training data from common drugs"""
    common_drugs = [
        "aspirin", "ibuprofen", "paracetamol", "amoxicillin", "atorvastatin",
        "metformin", "lisinopril", "levothyroxine", "amlodipine", "omeprazole",
        "simvastatin", "albuterol", "metoprolol", "prednisone", "azithromycin"
    ]
    
    training_data = []
    
    for drug in common_drugs:
        try:
            # Get FDA data for the drug
            response = requests.get(
                f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{drug}&limit=1",
                timeout=5
            )
            data = response.json().get("results", [None])[0]
            
            if data:
                # Extract features
                side_effects = extract_side_effects(data)
                preg_cat = data.get("pregnancy_category", ["C"])[0] if isinstance(
                    data.get("pregnancy_category"), list) else data.get("pregnancy_category", "C")
                
                # Calculate realistic rating
                rating = calculate_realistic_rating(len(side_effects), preg_cat)
                
                training_data.append({
                    'side_effects_count': len(side_effects),
                    'pregnancy_category_num': preg_map.get(preg_cat, 3),
                    'rating': rating
                })
                
        except Exception as e:
            print(f"Error processing {drug}: {e}")
            continue
    
    # Add some default training examples
    default_examples = [
        {'side_effects_count': 2, 'pregnancy_category_num': 1, 'rating': 9.5},  # Very safe
        {'side_effects_count': 5, 'pregnancy_category_num': 2, 'rating': 8.0},  # Safe
        {'side_effects_count': 8, 'pregnancy_category_num': 3, 'rating': 6.0},  # Moderate
        {'side_effects_count': 15, 'pregnancy_category_num': 4, 'rating': 3.0}, # Risky
        {'side_effects_count': 10, 'pregnancy_category_num': 5, 'rating': 1.5}, # Very risky
    ]
    training_data.extend(default_examples)
    
    return training_data

def calculate_realistic_rating(side_effects_count, preg_cat):
    """Calculate realistic rating between 1-10"""
    rating = 5.0  # Base rating
    
    # Pregnancy category adjustment
    preg_adjustment = {'A': +2, 'B': +1, 'C': 0, 'D': -2, 'X': -4}
    rating += preg_adjustment.get(preg_cat, 0)
    
    # Side effects adjustment
    if side_effects_count < 5:
        rating += 2
    elif side_effects_count < 10:
        rating += 1
    elif side_effects_count > 20:
        rating -= 2
    elif side_effects_count > 40:
        rating -= 3
    
    return max(1.0, min(10.0, rating))

def create_fallback_model():
    """Create a reasonable fallback model"""
    from sklearn.linear_model import LinearRegression
    
    # Realistic training data
    X = np.array([
        [2, 1], [5, 1], [3, 2], [8, 2], 
        [10, 3], [15, 3], [12, 4], [20, 5]
    ])
    y = np.array([9.5, 8.5, 8.0, 7.0, 5.5, 4.0, 3.0, 1.0])
    
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "rating_model.pkl")
    print("Fallback model created")
    return model

# KEEP ALL YOUR EXISTING FUNCTIONS EXACTLY THE SAME FROM HERE ↓

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
        # Ensure rating is between 1-10
        scaled_rating = max(1.0, min(10.0, rating))
        return {
            'Predicted Rating': round(scaled_rating,1),
            'Normal User Risk': 'Low' if scaled_rating>=8 else 'Moderate' if scaled_rating>=5 else 'High',
            'Pregnant Woman Risk': {
                'A':'Safe', 'B':'Likely Safe', 'C':'Caution',
                'D':'Unsafe', 'X':'Contraindicated'}.get(preg_cat,'Unknown'),
            'Pregnancy Category': preg_cat
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def get_drug_image(drug_name):
    """Get drug packaging image from multiple sources"""
    try:
        # Try multiple sources in order
        image_url = _get_image_from_drugs_com(drug_name)
        if image_url:
            return image_url
            
        image_url = _get_image_from_pharmacy_sites(drug_name)
        if image_url:
            return image_url
            
        image_url = _get_image_from_wikipedia(drug_name)
        if image_url:
            return image_url
            
        # Fallback to your original method
        url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=pageimages&piprop=original&titles={drug_name}"
        resp = requests.get(url, timeout=10).json()
        pages = resp["query"]["pages"]
        for _, p in pages.items():
            if "original" in p:
                return p["original"]["source"]
                
        return None
        
    except Exception as e:
        print(f"Image search error: {e}")
        return None

def _get_image_from_drugs_com(drug_name):
    """Try to get medicine packaging/box images from Drugs.com"""
    try:
        search_url = f"https://www.drugs.com/search.php?searchterm={drug_name.replace(' ', '+')}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for drug images in search results
        img_tags = soup.find_all('img', src=True)
        
        for img in img_tags:
            src = img.get('src', '')
            alt_text = img.get('alt', '').lower()
            img_lower = src.lower()
            
            # Skip logos and icons
            if any(logo in img_lower for logo in ['logo', 'icon', 'sprite', 'social']):
                continue
            
            # Look for packaging/box images (these often have specific patterns)
            if any(keyword in img_lower for keyword in ['/box/', '/packaging/', '/strip/', '/blister/', '/carton/', '/package/']):
                if src.startswith('//'):
                    return 'https:' + src
                elif src.startswith('/'):
                    return 'https://www.drugs.com' + src
                elif src.startswith('http'):
                    return src
        
        # If no packaging found, try Google Images for packaging specifically
        return _get_packaging_from_google(drug_name)
                
    except Exception as e:
        print(f"Drugs.com packaging image error: {e}")
    return None

def _get_packaging_from_google(drug_name):
    """Search Google for medicine packaging/box images specifically"""
    try:
        # Search specifically for packaging/box images
        search_url = f"https://www.google.com/search?q={drug_name.replace(' ', '+')}+medicine+box+packaging+strip&tbm=isch"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find image results
        img_tags = soup.find_all('img')
        for img in img_tags:
            src = img.get('src', '')
            if src.startswith('http') and not src.startswith('https://www.google.com'):
                # Prioritize images that look like packaging
                if any(keyword in src.lower() for keyword in ['box', 'strip', 'pack', 'blister', 'carton']):
                    return src
                return src  # Fallback to any image
                
    except Exception as e:
        print(f"Google packaging image error: {e}")
    return None

def _get_image_from_wikipedia(drug_name):
    """Try to get image from Wikipedia"""
    try:
        # Search for drug page
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={drug_name} drug&format=json"
        response = requests.get(search_url, timeout=10).json()
        
        if response['query']['search']:
            page_title = response['query']['search'][0]['title']
            
            # Get page image
            image_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={page_title}&prop=pageimages&format=json&pithumbsize=500"
            response = requests.get(image_url, timeout=10).json()
            
            pages = response.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if 'thumbnail' in page_data:
                    return page_data['thumbnail']['source']
                    
    except Exception as e:
        print(f"Wikipedia image error: {e}")
    return None

def _get_image_from_google(drug_name):
    """Try to get image from Google search"""
    try:
        search_url = f"https://www.google.com/search?q={drug_name.replace(' ', '+')}+pill+tablet+image&tbm=isch"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find image results
        img_tags = soup.find_all('img')
        for img in img_tags:
            src = img.get('src', '')
            if src.startswith('http') and not src.startswith('https://www.google.com'):
                return src
                
    except Exception as e:
        print(f"Google image error: {e}")
    return None 

def _get_image_from_pharmacy_sites(drug_name):
    """Try Indian pharmacy sites that often show medicine packaging"""
    try:
        # Try PharmEasy
        search_url = f"https://pharmeasy.in/search/all?name={drug_name.replace(' ', '%20')}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for product images (usually packaging)
        img_tags = soup.find_all('img', src=True)
        for img in img_tags:
            src = img.get('src', '')
            if 'product' in src.lower() or 'medicine' in src.lower():
                if src.startswith('//'):
                    return 'https:' + src
                elif src.startswith('/'):
                    return 'https://pharmeasy.in' + src
                return src
                
    except Exception as e:
        print(f"Pharmacy site image error: {e}")
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