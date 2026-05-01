from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import numpy as np
import joblib
import time

app = Flask(__name__)
app.secret_key = "traffic_secret_key"

# ===============================
# LOAD MODEL
# ===============================

model = joblib.load("traffic_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ===============================
# LOGIN
# ===============================

USERNAME = "admin"
PASSWORD = "admin"

# ===============================
# ADAPTIVE TIMING FUNCTION
# ===============================

def adaptive_green_time(total_vehicles):
    """
    Adaptive timing based on vehicle density
    """
    base_time = 20
    weight = 0.5
    max_time = 120

    green_time = base_time + (total_vehicles * weight)

    return min(int(green_time), max_time)

YELLOW_TIME = 5

# ===============================
# ROUTES
# ===============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/loginpage")
def loginpage():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    if request.form["username"] == USERNAME and request.form["password"] == PASSWORD:
        session["user"] = USERNAME
        return redirect(url_for("home"))
    return render_template("login.html", error="Invalid Credentials")

@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("loginpage"))
    return render_template("home.html")

# =========================================
# PREDICT WITH DASHBOARD RESULT
# =========================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if "user" not in session:
        return redirect(url_for("loginpage"))

    if request.method == "POST":

        junction_results = []

        for i in range(1, 5):

            car = int(request.form[f"car{i}"])
            bike = int(request.form[f"bike{i}"])
            bus = int(request.form[f"bus{i}"])
            truck = int(request.form[f"truck{i}"])

            total = car + bike + bus + truck

            features = np.array([[car, bike, bus, truck]])
            features_scaled = scaler.transform(features)

            pred_num = model.predict(features_scaled)[0]
            pred_label = label_encoder.inverse_transform([pred_num])[0]

            green_time = adaptive_green_time(total)

            junction_results.append({
                "junction": f"Junction {i}",
                "label": pred_label.capitalize(),
                "total": total,
                "green_time": green_time
            })

        # Priority sorting by total vehicles (realistic)
        junction_results.sort(key=lambda x: x["total"], reverse=True)

        session["junction_data"] = junction_results

        return redirect(url_for("dashboard"))

    return render_template("predict.html")


# =========================================
# DASHBOARD PAGE (POPUP RESULTS)
# =========================================

@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect(url_for("loginpage"))

    data = session.get("junction_data")

    if not data:
        return redirect(url_for("predict"))

    return render_template("dashboard.html", data=data)


# =========================================
# SIMULATION PAGE
# =========================================

@app.route("/simulation")
def simulation():

    if "user" not in session:
        return redirect(url_for("loginpage"))

    return render_template("simulation.html")


# =========================================
# API FOR REAL-TIME ROTATION
# =========================================

@app.route("/api/signal_sequence")
def signal_sequence():

    if "junction_data" not in session:
        return jsonify({"error": "No Data"}), 400

    data = session["junction_data"]

    # Ensure only first 4 junctions
    data = data[:4]

    # Attach index for frontend mapping
    for idx, j in enumerate(data):
        j["index"] = idx

    return jsonify({
        "sequence": data,
        "yellow_time": YELLOW_TIME,
        "total_junctions": 4,
        "generated_at": time.time()
    })
# =====================================
# RESET ROUTE
# =====================================

@app.route("/reset")
def reset():
    session.pop("junction_data", None)
    return redirect(url_for("predict"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("loginpage"))


if __name__ == "__main__":
    app.run(debug=True)