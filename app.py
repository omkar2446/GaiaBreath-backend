import os
import random
import sqlite3
import time
from datetime import datetime, timezone
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, Response, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from werkzeug.security import check_password_hash, generate_password_hash

from ai.rules.rule_engine import interpret_risk

app = Flask(__name__)
CORS(app)

# =========================
# GaiaBreath ML model
# =========================
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "community.db"
UPLOAD_DIR = ROOT / "uploads"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ADMIN_EMAIL = "otambe655@gmail.com"
ADMIN_PASSWORD = "12345678"
ADMIN_SESSION_TTL_SECONDS = 60 * 60 * 8
ADMIN_SESSIONS = {}
USER_SESSION_TTL_SECONDS = 60 * 60 * 24 * 7
USER_SESSIONS = {}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    UPLOAD_DIR.mkdir(exist_ok=True)
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS community_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                image_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def serialize_post(row):
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "image_url": row["image_path"],
        "created_at": row["created_at"],
    }


init_db()


def get_admin_token():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return auth.replace("Bearer ", "", 1).strip()


def get_user_token():
    return get_admin_token()


def is_admin_authenticated():
    token = get_admin_token()
    if not token:
        return False

    expires_at = ADMIN_SESSIONS.get(token)
    if not expires_at:
        return False
    if expires_at < time.time():
        ADMIN_SESSIONS.pop(token, None)
        return False
    return True


def create_user_session(user_id, email, name):
    token = uuid4().hex
    USER_SESSIONS[token] = {
        "user_id": user_id,
        "email": email,
        "name": name,
        "expires_at": time.time() + USER_SESSION_TTL_SECONDS,
    }
    return token


def get_authenticated_user():
    token = get_user_token()
    if not token:
        return None, None

    session = USER_SESSIONS.get(token)
    if not session:
        return None, None

    if session["expires_at"] < time.time():
        USER_SESSIONS.pop(token, None)
        return None, None

    return token, session


def load_if_exists(path):
    full_path = ROOT / path
    if full_path.exists():
        return joblib.load(full_path)
    return None


model = load_if_exists("ai/models/model.pkl")
scaler = load_if_exists("ai/models/scaler.pkl")

# =========================
# AQI personalized model
# =========================
aqi_model = load_if_exists("health_risk_model.pkl")
aqi_le = load_if_exists("label_encoder.pkl")


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status":"ok","message":"GaiaBreath AI running"})


@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.json or {}
    email = (data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")

    if email != ADMIN_EMAIL or password != ADMIN_PASSWORD:
        return jsonify({"error": "Invalid admin email or password"}), 401

    token = uuid4().hex
    ADMIN_SESSIONS[token] = time.time() + ADMIN_SESSION_TTL_SECONDS
    return jsonify({
        "token": token,
        "role": "admin",
        "admin_email": ADMIN_EMAIL,
        "expires_in_seconds": ADMIN_SESSION_TTL_SECONDS,
    })


@app.route("/api/auth/signup", methods=["POST"])
def auth_signup():
    data = request.json or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not email or "@" not in email:
        return jsonify({"error": "valid email is required"}), 400
    if len(password) < 8:
        return jsonify({"error": "password must be at least 8 characters"}), 400
    if email == ADMIN_EMAIL:
        return jsonify({"error": "This email is reserved for admin"}), 400

    created_at = datetime.now(timezone.utc).isoformat()
    password_hash = generate_password_hash(password)

    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO users (name, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, email, password_hash, created_at),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered"}), 409

    return jsonify({"status": "created", "email": email}), 201


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.json or {}
    email = (data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        token = uuid4().hex
        ADMIN_SESSIONS[token] = time.time() + ADMIN_SESSION_TTL_SECONDS
        return jsonify({
            "token": token,
            "role": "admin",
            "email": ADMIN_EMAIL,
            "name": "Admin",
            "expires_in_seconds": ADMIN_SESSION_TTL_SECONDS,
        })

    with get_db() as conn:
        row = conn.execute(
            """
            SELECT id, name, email, password_hash
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

    if row is None or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_user_session(
        user_id=row["id"],
        email=row["email"],
        name=row["name"],
    )
    return jsonify({
        "token": token,
        "role": "user",
        "email": row["email"],
        "name": row["name"],
        "expires_in_seconds": USER_SESSION_TTL_SECONDS,
    })


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    token = get_admin_token()
    if token and token in ADMIN_SESSIONS:
        if ADMIN_SESSIONS[token] >= time.time():
            return jsonify({"authenticated": True, "role": "admin", "email": ADMIN_EMAIL, "name": "Admin"})
        ADMIN_SESSIONS.pop(token, None)

    _, session = get_authenticated_user()
    if session:
        return jsonify({
            "authenticated": True,
            "role": "user",
            "email": session["email"],
            "name": session["name"],
        })

    return jsonify({"authenticated": False}), 401


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    token = get_admin_token()
    if not token:
        return jsonify({"error": "Authorization token required"}), 400

    removed = False
    if token in ADMIN_SESSIONS:
        ADMIN_SESSIONS.pop(token, None)
        removed = True
    if token in USER_SESSIONS:
        USER_SESSIONS.pop(token, None)
        removed = True

    if not removed:
        return jsonify({"status": "already_logged_out"})
    return jsonify({"status": "logged_out"})


# =========================
# OLD endpoint
# =========================
@app.route("/api/ai/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Legacy AI model is unavailable"}), 503

    data = request.json or {}

    try:
        aqi = float(data["aqi"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "aqi, temperature and humidity are required"}), 400

    X = np.array([[aqi, temperature, humidity]])
    X_scaled = scaler.transform(X)

    risk_level = int(model.predict(X_scaled)[0])

    rule_data = interpret_risk(
        risk_level,
        aqi=aqi,
        temperature=temperature,
        humidity=humidity
    )

    return jsonify({
        "risk_level":risk_level,
        "risk_label":rule_data["label"],
        "health_effects":rule_data["health_effects"],
        "precautions":rule_data["precautions"],
        "outdoor_travel":rule_data["outdoor_travel"],
        "transport_recommendation":rule_data["transport"],
        "air_purifier_required":rule_data["purifier"]
    })


# =========================
# AQI PERSONALIZED
# =========================
def adjust_risk(risk, age):
    levels = ["Low", "Mild", "Moderate", "High", "Severe"]

    if age < 12 or age > 50:
        if risk in levels and risk != "Severe":
            return levels[levels.index(risk) + 1]

    return risk


def suggestion(risk):
    if risk == "Low":
        return "Safe outdoor activity"
    if risk in ("Mild", "Moderate"):
        return "Limit outdoor exposure"
    if risk == "High":
        return "Avoid outdoor activity"
    if risk == "Severe":
        return "Stay indoors"
    return ""


def detailed_advice(risk):
    guidance = {
        "Low": {
            "description": "Air quality is acceptable for most people. Normal outdoor activity is generally safe.",
            "actions": [
                "Continue routine outdoor exercise and travel.",
                "Stay hydrated and monitor weather heat/humidity.",
                "Sensitive people should still watch for unusual symptoms.",
            ],
        },
        "Mild": {
            "description": "Air quality is slightly polluted. Sensitive groups may feel minor breathing discomfort.",
            "actions": [
                "Reduce intense outdoor workouts during peak traffic hours.",
                "Keep windows closed near busy roads in the evening.",
                "Children, elderly adults, and asthma patients should limit long exposure.",
            ],
        },
        "Moderate": {
            "description": "Air quality may affect sensitive people and can irritate eyes, throat, or breathing after prolonged exposure.",
            "actions": [
                "Limit outdoor time and choose lighter physical activity.",
                "Use a well-fitted mask (N95/KN95) when outdoors.",
                "Use indoor air cleaning and avoid smoke sources.",
            ],
        },
        "High": {
            "description": "Air quality is unhealthy. Breathing discomfort and fatigue are likely, especially for vulnerable people.",
            "actions": [
                "Avoid outdoor exercise and non-essential travel.",
                "Keep indoor air clean with purifier/filtration if available.",
                "Carry rescue medication if you have asthma/COPD and seek medical advice if symptoms worsen.",
            ],
        },
        "Severe": {
            "description": "Air quality is very unhealthy/hazardous. Health effects are possible for everyone.",
            "actions": [
                "Stay indoors as much as possible and seal indoor air leaks.",
                "Use N95/KN95 mask if you must step outside.",
                "Watch for chest tightness, persistent cough, dizziness, or breathlessness and seek urgent medical care when needed.",
            ],
        },
    }
    return guidance.get(
        risk,
        {
            "description": "Air quality guidance is unavailable for this risk label.",
            "actions": ["Monitor conditions and minimize exposure until updated guidance is available."],
        },
    )


@app.route("/api/aqi/predict", methods=["POST"])
def aqi_predict():
    if aqi_model is None or aqi_le is None:
        return jsonify({"error": "AQI model artifacts are unavailable"}), 503

    data = request.json or {}

    try:
        age = int(data["age"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "age is required and must be numeric"}), 400

    required = ["temperature", "humidity", "pm2_5", "pm10", "no2", "co", "so2"]
    missing = [key for key in required if key not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        features = {key: float(data[key]) for key in required}
    except (TypeError, ValueError):
        return jsonify({"error": "All pollutant and weather fields must be numeric"}), 400
    sample = pd.DataFrame([features])

    pred = aqi_model.predict(sample)
    base = aqi_le.inverse_transform(pred)[0]

    final = adjust_risk(base, age)
    advice = detailed_advice(final)

    return jsonify({
        "base_risk": base,
        "final_risk": final,
        "suggestion": suggestion(final),
        "description": advice["description"],
        "advice_actions": advice["actions"],
    })


# new endpoint that builds a simple textual health report
@app.route("/api/health/report", methods=["POST"])
def health_report():
    """Produce a personalized report from age, smoking habit, outdoor time, AQI and temperature."""
    data = request.json or {}

    try:
        age = int(data.get("age"))
    except (TypeError, ValueError):
        return jsonify({"error": "age is required and must be numeric"}), 400

    smoke_raw = data.get("smoke")
    if isinstance(smoke_raw, bool):
        smoke = smoke_raw
    elif isinstance(smoke_raw, str):
        smoke = smoke_raw.strip().lower() in {"yes", "true", "1", "smoker"}
    else:
        return jsonify({"error": "smoke is required and must be yes/no or boolean"}), 400

    try:
        outdoor_hours = float(data.get("outdoor_hours"))
    except (TypeError, ValueError):
        return jsonify({"error": "outdoor_hours is required and must be numeric"}), 400
    try:
        avg_aqi = float(data.get("avg_aqi"))
    except (TypeError, ValueError):
        return jsonify({"error": "avg_aqi is required and must be numeric"}), 400
    try:
        avg_temp = float(data.get("avg_temp"))
    except (TypeError, ValueError):
        return jsonify({"error": "avg_temp is required and must be numeric"}), 400

    if age < 1 or age > 110:
        return jsonify({"error": "age must be between 1 and 110"}), 400
    if outdoor_hours < 0 or outdoor_hours > 24:
        return jsonify({"error": "outdoor_hours must be between 0 and 24"}), 400
    if avg_aqi < 0 or avg_aqi > 1000:
        return jsonify({"error": "avg_aqi must be between 0 and 1000"}), 400
    if avg_temp < -30 or avg_temp > 60:
        return jsonify({"error": "avg_temp must be between -30 and 60"}), 400

    score = 0

    if age >= 60:
        score += 3
    elif age >= 45:
        score += 2
    elif age >= 30:
        score += 1

    if smoke:
        score += 3

    if outdoor_hours >= 8:
        score += 3
    elif outdoor_hours >= 5:
        score += 2
    elif outdoor_hours >= 2:
        score += 1

    if avg_aqi >= 301:
        score += 4
    elif avg_aqi >= 201:
        score += 3
    elif avg_aqi >= 151:
        score += 2
    elif avg_aqi >= 101:
        score += 1

    if avg_temp >= 40 or avg_temp <= 0:
        score += 2
    elif avg_temp >= 35 or avg_temp <= 5:
        score += 1

    if score <= 3:
        risk = "Low"
        summary = "Your current profile suggests low respiratory stress risk."
        actions = [
            "Continue normal activity with hydration and regular breaks.",
            "Avoid heavy traffic routes when possible.",
            "Maintain a healthy sleep routine.",
        ]
    elif score <= 6:
        risk = "Mild"
        summary = "Your profile suggests mild risk with occasional sensitivity."
        actions = [
            "Reduce prolonged time in polluted outdoor locations.",
            "Prefer morning or low-traffic hours for outdoor plans.",
            "Consider basic mask use in heavy traffic.",
        ]
    elif score <= 9:
        risk = "Moderate"
        summary = "Your profile suggests moderate respiratory risk."
        actions = [
            "Use N95/KN95 mask for longer outdoor exposure.",
            "Take shorter outdoor sessions with indoor recovery breaks.",
            "Use indoor air filtration where possible.",
        ]
    elif score <= 12:
        risk = "High"
        summary = "Your profile suggests high risk and needs active precautions."
        actions = [
            "Avoid non-essential prolonged outdoor exposure.",
            "Strictly use N95/KN95 in polluted areas.",
            "If smoking, start a reduction/cessation plan immediately.",
        ]
    else:
        risk = "Severe"
        summary = "Your profile suggests severe risk and immediate behavior changes are advised."
        actions = [
            "Strongly limit outdoor time to essential travel only.",
            "Use high-quality masks and keep indoor air clean.",
            "Seek clinical advice for personalized lung-health screening.",
        ]

    smoke_text = "Yes" if smoke else "No"
    report_text = (
        f"Health Report (UTC {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M')}):\n"
        f"Age: {age}\n"
        f"Smoker: {smoke_text}\n"
        f"Outdoor time per day: {outdoor_hours:.1f} hours\n"
        f"Average AQI: {avg_aqi:.1f}\n"
        f"Average Temperature: {avg_temp:.1f} C\n"
        f"Risk score: {score}/16\n"
        f"Risk level: {risk}\n"
        f"Summary: {summary}\n"
        "Recommended actions:\n"
        f"- {actions[0]}\n"
        f"- {actions[1]}\n"
        f"- {actions[2]}"
    )

    return jsonify({
        "age": age,
        "smoke": smoke,
        "outdoor_hours": outdoor_hours,
        "avg_aqi": avg_aqi,
        "avg_temp": avg_temp,
        "risk_score": score,
        "risk_level": risk,
        "summary": summary,
        "actions": actions,
        "report": report_text,
    })


def aqi_band(aqi):
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


@app.route("/api/iot/report", methods=["GET"])
def iot_report():
    aqi = random.randint(25, 360)
    temperature = round(random.uniform(16.0, 43.0), 1)
    humidity = random.randint(25, 95)
    trend = random.choice(["rising", "falling", "stable"])
    node = (request.args.get("node") or "NODE-01").strip() or "NODE-01"

    return jsonify({
        "node": node,
        "aqi": aqi,
        "aqi_band": aqi_band(aqi),
        "temperature": temperature,
        "humidity": humidity,
        "trend": trend,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "refresh_hint_seconds": 2,
    })


@app.route("/api/aqi/map/tile/<int:z>/<int:x>/<int:y>.png", methods=["GET"])
def aqi_map_tile_proxy(z, x, y):
    if z < 0 or z > 13:
        return jsonify({"error": "zoom level must be between 0 and 13"}), 400
    if x < 0 or y < 0:
        return jsonify({"error": "tile coordinates must be non-negative"}), 400

    layers = {"usepa-aqi", "aqi", "pm25", "wind"}
    layer = (request.args.get("layer") or "usepa-aqi").strip().lower()
    if layer not in layers:
        return jsonify({"error": f"Unsupported layer. Allowed: {', '.join(sorted(layers))}"}), 400

    # Keep X wrapped for world maps while Y remains clamped to valid tile rows.
    n = 2 ** z
    x = x % n
    if y > n - 1:
        return jsonify({"error": "tile y is outside valid range for this zoom"}), 400

    token = (os.getenv("WAQI_TOKEN") or "demo").strip()
    upstream_url = f"https://tiles.waqi.info/tiles/{layer}/{z}/{x}/{y}.png?token={token}"
    upstream_req = Request(upstream_url, headers={"User-Agent": "GaiaBreathAI/1.0"})

    try:
        with urlopen(upstream_req, timeout=10) as upstream:
            tile = upstream.read()
        resp = Response(tile, mimetype="image/png")
        resp.headers["Cache-Control"] = "public, max-age=300"
        return resp
    except HTTPError as exc:
        return jsonify({"error": f"Upstream tile service error ({exc.code})"}), 502
    except URLError:
        return jsonify({"error": "Could not reach AQI tile provider"}), 502


@app.route("/api/map/base/tile/<int:z>/<int:x>/<int:y>.png", methods=["GET"])
def base_map_tile_proxy(z, x, y):
    if z < 0 or z > 19:
        return jsonify({"error": "zoom level must be between 0 and 19"}), 400
    if x < 0 or y < 0:
        return jsonify({"error": "tile coordinates must be non-negative"}), 400

    n = 2 ** z
    x = x % n
    if y > n - 1:
        return jsonify({"error": "tile y is outside valid range for this zoom"}), 400

    upstream_url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    upstream_req = Request(upstream_url, headers={"User-Agent": "GaiaBreathAI/1.0"})

    try:
        with urlopen(upstream_req, timeout=10) as upstream:
            tile = upstream.read()
        resp = Response(tile, mimetype="image/png")
        resp.headers["Cache-Control"] = "public, max-age=3600"
        return resp
    except HTTPError as exc:
        return jsonify({"error": f"Upstream map tile service error ({exc.code})"}), 502
    except URLError:
        return jsonify({"error": "Could not reach base map tile provider"}), 502


@app.route("/api/community/posts", methods=["GET"])
def list_community_posts():
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, name, description, image_path, created_at
            FROM community_posts
            ORDER BY id DESC
            """
        ).fetchall()

    return jsonify({"posts": [serialize_post(row) for row in rows]})


@app.route("/api/community/posts", methods=["POST"])
def create_community_post():
    name = (request.form.get("name") or "").strip()
    description = (request.form.get("description") or "").strip()
    photo = request.files.get("photo")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not description:
        return jsonify({"error": "description is required"}), 400
    if photo is None or not photo.filename:
        return jsonify({"error": "photo is required"}), 400

    ext = Path(photo.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        return jsonify({"error": f"photo type must be one of: {allowed}"}), 400

    # Store with random filename to avoid conflicts and unsafe paths.
    random_name = f"{uuid4().hex}{ext}"
    target_path = UPLOAD_DIR / random_name
    photo.save(target_path)

    image_path = f"/uploads/{random_name}"
    created_at = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO community_posts (name, description, image_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (name, description, image_path, created_at),
        )
        conn.commit()
        post_id = cur.lastrowid

        row = conn.execute(
            """
            SELECT id, name, description, image_path, created_at
            FROM community_posts
            WHERE id = ?
            """,
            (post_id,),
        ).fetchone()

    return jsonify(serialize_post(row)), 201


@app.route("/api/community/posts/<int:post_id>", methods=["DELETE"])
def delete_community_post(post_id):
    if not is_admin_authenticated():
        return jsonify({"error": "Admin authentication required"}), 401

    with get_db() as conn:
        row = conn.execute(
            """
            SELECT id, image_path
            FROM community_posts
            WHERE id = ?
            """,
            (post_id,),
        ).fetchone()

        if row is None:
            return jsonify({"error": "Post not found"}), 404

        conn.execute(
            """
            DELETE FROM community_posts
            WHERE id = ?
            """,
            (post_id,),
        )
        conn.commit()

    image_path = row["image_path"] or ""
    if image_path.startswith("/uploads/"):
        filename = image_path.replace("/uploads/", "", 1)
        target = UPLOAD_DIR / filename
        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass

    return jsonify({"status": "deleted", "id": post_id})


if __name__ == "__main__":
    app.run(debug=True)

