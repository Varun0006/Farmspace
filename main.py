from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import urlopen

from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from model import MODEL_PATH, load_trained_model, suggest_best_crop, train_model


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "farmspace.db"

SOIL_TYPES = ["Sandy", "Loamy", "Clay", "Silty", "Peaty"]
WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Stormy"]
CROP_TYPES = ["Wheat", "Rice", "Corn", "Barley", "Soybeans"]

PRODUCT_PLACEHOLDER = "images/product-placeholder.svg"
PRODUCTS = [
    (
        "Neem Shield",
        "Pesticide",
        185.0,
        "Organic pest control spray for common leaf-eating insects.",
        "images/neem-shield.jpg",
    ),
    (
        "CropGuard Plus",
        "Pesticide",
        240.0,
        "Broad-spectrum protection for small farm applications.",
        "images/crop-guard.jpg",
    ),
    (
        "Urea Granules",
        "Fertilizer",
        225.0,
        "High-nitrogen fertilizer commonly used for fast vegetative growth.",
        "images/urea.jpg",
    ),
    (
        "Ammonium Sulfate",
        "Fertilizer",
        240.0,
        "Nitrogen and sulfur source that helps support greener leaves.",
        "images/Ammonium-Sulphate.png",
    ),
    (
        "DAP",
        "Fertilizer",
        295.0,
        "Diammonium phosphate fertilizer for strong root and early crop development.",
        "images/dap.jpg",
    ),
    (
        "MOP",
        "Fertilizer",
        265.0,
        "Muriate of potash for improving crop quality and water regulation.",
        "images/mop.jpg",
    ),
    (
        "SSP",
        "Fertilizer",
        237.5,
        "Single super phosphate to support root growth and flowering.",
        "images/ssp.jpg",
    ),
    (
        "Calcium Nitrate",
        "Fertilizer",
        332.5,
        "Fast-acting fertilizer for calcium and nitrogen supply.",
        "images/Calcium-Nitrate.webp",
    ),
    (
        "Root Boost",
        "Fertilizer",
        272.5,
        "Root-development fertilizer designed for early crop vigor.",
        "images/root-boost-banner-left.png",
    ),
]

PRODUCT_BY_NAME = {product[0]: product for product in PRODUCTS}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-farmspace-secret-key")


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        connection = sqlite3.connect(DATABASE_PATH)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        g.db = connection
    return g.db


@app.teardown_appcontext
def close_db(_: BaseException | None) -> None:
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db() -> None:
    database = get_db()
    database.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            description TEXT NOT NULL,
            image_path TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            total_amount REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            subtotal REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders (id) ON DELETE CASCADE,
            FOREIGN KEY (product_id) REFERENCES products (id)
        );
        """
    )
    database.commit()


def seed_products() -> None:
    database = get_db()
    existing_rows = database.execute(
        "SELECT id, name FROM products"
    ).fetchall()
    existing_names = {row["name"] for row in existing_rows}
    canonical_names = {product[0] for product in PRODUCTS}

    missing_products = [product for product in PRODUCTS if product[0] not in existing_names]
    if missing_products:
        database.executemany(
            """
            INSERT INTO products (name, category, price, description, image_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            missing_products,
        )

    for product in PRODUCTS:
        database.execute(
            """
            UPDATE products
            SET category = ?, price = ?, description = ?, image_path = ?
            WHERE name = ?
            """,
            (product[1], product[2], product[3], product[4], product[0]),
        )

    stale_names = existing_names - canonical_names
    if stale_names:
        placeholders = ",".join("?" for _ in stale_names)
        database.execute(
            f"DELETE FROM products WHERE name IN ({placeholders})",
            tuple(stale_names),
        )
    database.commit()


def ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    train_model()


def current_user() -> dict[str, Any] | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    database = get_db()
    user = database.execute(
        "SELECT id, name, email FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    return dict(user) if user else None


def cart_count() -> int:
    cart = session.get("cart", {})
    return int(sum(cart.values()))


@app.context_processor
def inject_globals() -> dict[str, Any]:
    return {
        "current_user": current_user(),
        "cart_count": cart_count(),
    }


def login_required(view: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(view)
    def wrapped_view(*args: Any, **kwargs: Any) -> Any:
        if not session.get("user_id"):
            flash("Please log in to access that page.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def get_cart_items() -> list[dict[str, Any]]:
    cart = session.get("cart", {})
    if not cart:
        return []

    product_ids = [int(product_id) for product_id in cart.keys()]
    placeholders = ",".join("?" for _ in product_ids)
    database = get_db()
    products = database.execute(
        f"SELECT id, name, category, price, description, image_path FROM products WHERE id IN ({placeholders})",
        product_ids,
    ).fetchall()

    items: list[dict[str, Any]] = []
    for product in products:
        quantity = int(cart.get(str(product["id"]), 0))
        item = dict(product)
        item["quantity"] = quantity
        item["subtotal"] = float(product["price"]) * quantity
        items.append(item)
    return items


def create_order_from_cart(user_id: int) -> int:
    items = get_cart_items()
    if not items:
        raise ValueError("Cart is empty.")

    total_amount = float(sum(item["subtotal"] for item in items))
    database = get_db()

    cursor = database.execute(
        "INSERT INTO orders (user_id, total_amount) VALUES (?, ?)",
        (user_id, total_amount),
    )
    order_id = int(cursor.lastrowid)

    database.executemany(
        """
        INSERT INTO order_items (order_id, product_id, quantity, unit_price, subtotal)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                order_id,
                int(item["id"]),
                int(item["quantity"]),
                float(item["price"]),
                float(item["subtotal"]),
            )
            for item in items
        ],
    )
    database.commit()

    return order_id


def get_order_with_items(order_id: int, user_id: int) -> dict[str, Any] | None:
    database = get_db()
    order = database.execute(
        """
        SELECT id, user_id, total_amount, created_at
        FROM orders
        WHERE id = ? AND user_id = ?
        """,
        (order_id, user_id),
    ).fetchone()
    if order is None:
        return None

    items = database.execute(
        """
        SELECT
            oi.product_id,
            oi.quantity,
            oi.unit_price,
            oi.subtotal,
            p.name,
            p.category,
            p.image_path
        FROM order_items oi
        INNER JOIN products p ON p.id = oi.product_id
        WHERE oi.order_id = ?
        ORDER BY oi.id ASC
        """,
        (order_id,),
    ).fetchall()

    return {
        "order": dict(order),
        "items": [dict(item) for item in items],
    }


def map_weather_code_to_condition(weather_code: int) -> str:
    if weather_code in (0,):
        return "Sunny"
    if weather_code in (1, 2, 3, 45, 48):
        return "Cloudy"
    if weather_code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86):
        return "Rainy"
    if weather_code in (95, 96, 99):
        return "Stormy"
    return "Cloudy"


def fetch_weather_by_coordinates(latitude: float, longitude: float) -> dict[str, Any]:
    query = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
            "daily": "precipitation_sum",
            "forecast_days": 1,
            "timezone": "auto",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{query}"

    with urlopen(url, timeout=10) as response:
        payload = response.read().decode("utf-8")

    import json

    data = json.loads(payload)
    current = data.get("current", {})
    daily = data.get("daily", {})
    daily_precip_list = daily.get("precipitation_sum", [])

    temperature = float(current.get("temperature_2m", 0.0))
    humidity = float(current.get("relative_humidity_2m", 0.0))
    rainfall = (
        float(daily_precip_list[0])
        if daily_precip_list
        else float(current.get("precipitation", 0.0))
    )
    weather_code = int(current.get("weather_code", 3))

    return {
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "weather_condition": map_weather_code_to_condition(weather_code),
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }


def map_wrb_to_app_soil(wrb_name: str) -> str:
    wrb = wrb_name.lower()

    sandy_markers = ("arenosol", "regosol")
    clay_markers = ("vertisol", "nitisol", "luvisol", "acrisol", "alisol")
    silty_markers = ("fluvisol", "gleysol", "planosol")
    peaty_markers = ("histosol",)
    loamy_markers = (
        "cambisol",
        "phaeozem",
        "chernozem",
        "kastanozem",
        "umbrisol",
        "andosol",
    )

    if any(marker in wrb for marker in peaty_markers):
        return "Peaty"
    if any(marker in wrb for marker in clay_markers):
        return "Clay"
    if any(marker in wrb for marker in sandy_markers):
        return "Sandy"
    if any(marker in wrb for marker in silty_markers):
        return "Silty"
    if any(marker in wrb for marker in loamy_markers):
        return "Loamy"
    return "Loamy"


def fetch_common_soil_type_by_coordinates(latitude: float, longitude: float) -> dict[str, Any]:
    query = urlencode(
        {
            "lat": latitude,
            "lon": longitude,
            "number_classes": 3,
        }
    )
    url = f"https://rest.isric.org/soilgrids/v2.0/classification/query?{query}"

    with urlopen(url, timeout=10) as response:
        payload = response.read().decode("utf-8")

    import json

    data = json.loads(payload)
    wrb_class_name = str(data.get("wrb_class_name", ""))
    wrb_class_probability = data.get("wrb_class_probability", [])

    return {
        "soil_type": map_wrb_to_app_soil(wrb_class_name),
        "soil_wrb_class": wrb_class_name,
        "soil_top_classes": wrb_class_probability,
    }


def format_weekday_label(date_text: str) -> str:
    return datetime.strptime(date_text, "%Y-%m-%d").strftime("%A")


def fetch_weekly_weather_by_coordinates(latitude: float, longitude: float) -> dict[str, Any]:
    query = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "forecast_days": 7,
            "timezone": "auto",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{query}"

    with urlopen(url, timeout=10) as response:
        payload = response.read().decode("utf-8")

    import json

    data = json.loads(payload)
    daily = data.get("daily", {})

    dates = daily.get("time", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    rainfall = daily.get("precipitation_sum", [])
    weather_codes = daily.get("weather_code", [])

    forecast: list[dict[str, Any]] = []
    for index, date_text in enumerate(dates):
        weather_code = int(weather_codes[index]) if index < len(weather_codes) else 3
        forecast.append(
            {
                "date": date_text,
                "day": format_weekday_label(date_text),
                "max_temp": float(max_temps[index]) if index < len(max_temps) else 0.0,
                "min_temp": float(min_temps[index]) if index < len(min_temps) else 0.0,
                "rainfall": float(rainfall[index]) if index < len(rainfall) else 0.0,
                "weather_condition": map_weather_code_to_condition(weather_code),
                "weather_code": weather_code,
            }
        )

    return {
        "forecast": forecast,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }


@app.route("/")
def home() -> str:
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register() -> Any:
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All registration fields are required.", "danger")
            return render_template("register.html")

        database = get_db()
        try:
            database.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, generate_password_hash(password)),
            )
            database.commit()
        except sqlite3.IntegrityError:
            flash("That email is already registered.", "danger")
            return render_template("register.html")

        flash("Registration complete. You can log in now.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login() -> Any:
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        database = get_db()
        user = database.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "danger")
            return render_template("login.html")

        session.clear()
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        session["user_email"] = user["email"]
        flash(f"Welcome back, {user['name']}.", "success")
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
def logout() -> Any:
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict() -> Any:
    recommendation = None
    crop_ranking: list[dict[str, Any]] = []
    form_data = {
        "temperature": "",
        "rainfall": "",
        "humidity": "",
        "soil_type": SOIL_TYPES[0],
        "weather_condition": WEATHER_CONDITIONS[0],
    }

    if request.method == "POST":
        form_data["temperature"] = request.form.get("temperature", "").strip()
        form_data["rainfall"] = request.form.get("rainfall", "").strip()
        form_data["humidity"] = request.form.get("humidity", "").strip()
        form_data["soil_type"] = request.form.get("soil_type", SOIL_TYPES[0])
        form_data["weather_condition"] = request.form.get("weather_condition", WEATHER_CONDITIONS[0])

        try:
            pipeline, _ = load_trained_model()
        except FileNotFoundError:
            pipeline, _ = train_model()

        try:
            recommendation = suggest_best_crop(
                pipeline,
                temperature=float(form_data["temperature"]),
                rainfall=float(form_data["rainfall"]),
                humidity=float(form_data["humidity"]),
                soil_type=form_data["soil_type"],
                weather_condition=form_data["weather_condition"],
                crop_types=CROP_TYPES,
            )
            crop_ranking = recommendation["ranking"]
        except ValueError:
            flash("Please enter valid numeric values for temperature, rainfall, and humidity.", "danger")

    return render_template(
        "prediction.html",
        recommendation=recommendation,
        crop_ranking=crop_ranking,
        soil_types=SOIL_TYPES,
        weather_conditions=WEATHER_CONDITIONS,
        form_data=form_data,
    )


@app.route("/api/weather/current")
@login_required
def weather_current() -> Any:
    try:
        latitude = float(request.args.get("lat", ""))
        longitude = float(request.args.get("lon", ""))
    except ValueError:
        return jsonify({"error": "Invalid latitude or longitude."}), 400

    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        return jsonify({"error": "Latitude or longitude out of range."}), 400

    try:
        weather_data = fetch_weather_by_coordinates(latitude=latitude, longitude=longitude)
    except Exception:
        return jsonify({"error": "Could not fetch weather for your location right now."}), 502

    try:
        soil_data = fetch_common_soil_type_by_coordinates(latitude=latitude, longitude=longitude)
    except Exception:
        soil_data = {
            "soil_type": "Loamy",
            "soil_wrb_class": "Unknown",
            "soil_top_classes": [],
        }

    return jsonify({**weather_data, **soil_data})


@app.route("/api/weather/weekly")
def weather_weekly_api() -> Any:
    try:
        latitude = float(request.args.get("lat", ""))
        longitude = float(request.args.get("lon", ""))
    except ValueError:
        return jsonify({"error": "Invalid latitude or longitude."}), 400

    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        return jsonify({"error": "Latitude or longitude out of range."}), 400

    try:
        weekly_data = fetch_weekly_weather_by_coordinates(latitude=latitude, longitude=longitude)
    except Exception:
        return jsonify({"error": "Could not fetch weekly weather for your location right now."}), 502

    return jsonify(weekly_data)


@app.route("/weather-report")
def weather_report() -> str:
    return render_template("weather_report.html")


@app.route("/shop")
def shop() -> str:
    database = get_db()
    products_rows = database.execute(
        "SELECT id, name, category, price, description, image_path FROM products ORDER BY category, name"
    ).fetchall()
    products = [dict(row) for row in products_rows]

    categories = ["Pesticide", "Fertilizer"]
    products_by_category: dict[str, list[dict[str, Any]]] = {
        category: [product for product in products if product["category"] == category]
        for category in categories
    }

    return render_template(
        "shop.html",
        products=products,
        categories=categories,
        products_by_category=products_by_category,
    )


@app.route("/add-to-cart/<int:product_id>", methods=["POST"])
@login_required
def add_to_cart(product_id: int) -> Any:
    database = get_db()
    product = database.execute(
        "SELECT id, name FROM products WHERE id = ?",
        (product_id,),
    ).fetchone()
    if product is None:
        flash("That product does not exist.", "danger")
        return redirect(url_for("shop"))

    quantity_raw = request.form.get("quantity", "1").strip()
    try:
        quantity = int(quantity_raw)
    except ValueError:
        flash("Quantity must be a valid number.", "danger")
        return redirect(url_for("shop"))

    if quantity < 1:
        flash("Quantity must be at least 1.", "warning")
        return redirect(url_for("shop"))
    if quantity > 25:
        flash("Maximum quantity per add is 25.", "warning")
        return redirect(url_for("shop"))

    cart = session.get("cart", {})
    key = str(product_id)
    cart[key] = int(cart.get(key, 0)) + quantity
    session["cart"] = cart
    session.modified = True
    flash(f"Added {quantity} x {product['name']} to your cart.", "success")
    return redirect(url_for("shop"))


@app.route("/remove-from-cart/<int:product_id>", methods=["POST"])
@login_required
def remove_from_cart(product_id: int) -> Any:
    cart = session.get("cart", {})
    key = str(product_id)
    if key in cart:
        quantity = int(cart[key]) - 1
        if quantity > 0:
            cart[key] = quantity
        else:
            cart.pop(key)
        session["cart"] = cart
        session.modified = True
        flash("Item updated in cart.", "info")
    return redirect(url_for("cart"))


@app.route("/cart")
@login_required
def cart() -> str:
    items = get_cart_items()
    total = sum(item["subtotal"] for item in items)
    return render_template("cart.html", items=items, total=total)


@app.route("/orders")
@login_required
def orders() -> str:
    database = get_db()
    user_id = int(session["user_id"])
    user_orders = database.execute(
        """
        SELECT id, total_amount, created_at
        FROM orders
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user_id,),
    ).fetchall()
    return render_template("orders.html", orders=user_orders)


@app.route("/orders/<int:order_id>")
@login_required
def order_detail(order_id: int) -> Any:
    user_id = int(session["user_id"])
    order_bundle = get_order_with_items(order_id=order_id, user_id=user_id)
    if order_bundle is None:
        flash("Order not found.", "warning")
        return redirect(url_for("orders"))

    return render_template(
        "order_detail.html",
        order=order_bundle["order"],
        items=order_bundle["items"],
    )


@app.route("/checkout", methods=["POST"])
@login_required
def checkout() -> Any:
    user_id = session.get("user_id")
    if not user_id:
        flash("Please log in to complete checkout.", "warning")
        return redirect(url_for("login"))

    if not session.get("cart"):
        flash("Your cart is empty.", "warning")
        return redirect(url_for("cart"))

    try:
        order_id = create_order_from_cart(user_id=int(user_id))
    except ValueError:
        flash("Your cart is empty.", "warning")
        return redirect(url_for("cart"))

    session.pop("cart", None)
    flash("Checkout successful. Your order has been placed.", "success")
    return redirect(url_for("order_detail", order_id=order_id))


def bootstrap_app() -> None:
    with app.app_context():
        init_db()
        seed_products()
        ensure_model()


bootstrap_app()


def main() -> None:
    app.run(debug=True)


if __name__ == "__main__":
    main()
