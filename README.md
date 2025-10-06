# Drug-Forecaster

🔍 Problem Statement
Can we accurately predict the effectiveness and rating of a drug using real-world data such as side effects and pregnancy category, fetched from trusted sources like the FDA?

🧪 How I Built It<img width="1100" height="620" alt="Screenshot 2025-08-03 073146" src="https://github.com/user-attachments/assets/60d3c281-b707-4a30-b7a6-78c3e266409d" />

[📥 Download Drug Forecaster Video](Drug%20ForecasterVideo.mp4)

📥 The video is too large to play directly on GitHub.  
You can download it by clicking the **Download** button on the top-right of this file.

🧬 FDA Integration
The model fetches real-time data from the official FDA (Food & Drug Administration) database using the requests library.

This ensures up-to-date details on:

Side effects

Usage warnings

Drug class

Pregnancy category

✅ This gives the model real-world relevance and trustworthiness.

📊 Dataset
Initially, I used an Excel dataset with:

Drug names

User reviews

Ratings (1 to 10)

Side effects

Pregnancy category

Then, I enriched this with live FDA data for better predictions.

🤖 Machine Learning Algorithm Used
✅ Random Forest Regressor
Why? Random Forest is an ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting.

Works great with mixed data types like numerical and categorical features.

It handles:

The number of side effects (quantified)

Pregnancy category (mapped to numbers)

Drug details from FDA

🧰 Python Libraries Used
Library	Purpose
pandas	Load and clean Excel dataset
numpy	Perform numerical operations
scikit-learn	Build and train ML model (RandomForestRegressor)
joblib	Save and load the trained model
Flask	Build the web application
requests	Fetch real-time drug data from FDA’s API
Jinja2	Display results dynamically in HTML
