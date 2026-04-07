# Farmspace

Farmspace is a Flask-based agriculture web app for crop yield prediction, farm supply shopping, and weather planning. It uses a trained machine learning model to suggest the best crop for the current conditions, stores users in SQLite, and supports a simple cart and order history flow.

## Features

- Crop recommendation and yield prediction from weather and soil inputs
- Automatic location-based weather and soil lookup
- Weekly weather report page
- User registration, login, and logout
- Product shop for pesticides and fertilizers
- Quantity-based cart
- Persistent checkout with order history and order details
- SQLite database for users, products, orders, and order items

## Tech Stack

- Python 3.14+
- Flask
- SQLite
- Pandas
- NumPy
- scikit-learn
- Joblib

## Project Structure

- `model.py` - ML training and prediction logic
- `main.py` - Flask application and CLI entrypoint; contains routes, database setup, weather APIs, cart, and orders
- `crop_yield_dataset.csv` - Training dataset
- `artifacts/` - Saved ML model artifacts
- `static/` - CSS and product images
- `templates/` - HTML templates

## Setup

1. Install dependencies:

	```bash
	uv sync
	```

2. Train the model and create the artifact:

	```bash
	uv run python main.py
	```

3. Run the Flask app:

	```bash
	uv run python main.py
	```

4. Open the app in your browser:

	```
	http://127.0.0.1:5000
	```

## Main Pages

- `/` - Home page
- `/register` - User registration
- `/login` - Login page
- `/predict` - Crop recommendation page
- `/weather-report` - Weekly weather report page
- `/shop` - Product shop
- `/cart` - Shopping cart
- `/orders` - Order history

## How the Model Works

The model is trained on the crop yield dataset using a Random Forest Regressor. It accepts:

- Temperature
- Rainfall
- Humidity
- Soil type
- Weather condition

It then evaluates the supported crops and recommends the one with the highest predicted yield.

## Data Sources

- Weather data is fetched from Open-Meteo
- Soil classification is fetched from SoilGrids

## Notes

- The app stores generated data locally in `farmspace.db` and `artifacts/`.
- These files are ignored by Git so the repository stays clean.
- If location access is blocked in the browser, weather and soil can still be entered manually.

## License

This project is intended for educational use and college-level demo work.
