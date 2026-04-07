# Farmspace

[![Python](https://img.shields.io/badge/Python-3.14+-4B8BBE?logo=python&logoColor=white)](https://www.python.org/)
[![UV](https://img.shields.io/badge/UV-Setup-111827?logo=python&logoColor=white)](https://docs.astral.sh/uv/)
[![Flask](https://img.shields.io/badge/Flask-3.1-111827?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-0C7BDC?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-4DABF7?logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-FBBF24?logo=scikitlearn&logoColor=111827)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-DB-0F766E?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Joblib](https://img.shields.io/badge/Joblib-Model-8B5CF6?logo=python&logoColor=white)](https://joblib.readthedocs.io/)
[![MIT License](https://img.shields.io/badge/License-MIT-EC4899?logo=open-source-initiative&logoColor=white)](LICENSE)

Farmspace is a Flask-based agriculture platform for crop yield prediction, farm supply shopping, and weather planning. It uses a trained machine learning model to recommend the best crop for the current conditions, stores users in SQLite, and supports a cart, checkout, and order history flow.

## Features

- Crop recommendation and yield prediction from weather and soil inputs
- Automatic location-based weather and soil lookup
- Weekly weather report page
- User registration, login, and logout
- Product shop for pesticides and fertilizers
- Quantity-based cart
- Persistent checkout with order history and order details
- SQLite database for users, products, orders, and order items

## Project Structure

- `model.py` - ML training and prediction logic
- `main.py` - Flask application and CLI entrypoint; contains routes, database setup, weather APIs, cart, and orders
- `crop_yield_dataset.csv` - Training dataset
- `artifacts/` - Saved ML model artifacts
- `static/` - CSS and product images
- `templates/` - HTML templates

## Quick Start

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

## License

This project is licensed under the [MIT License](LICENSE).
