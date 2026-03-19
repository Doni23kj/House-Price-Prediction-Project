import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

MODEL_PATH = "models/model.pkl"
DATA_PATH = "Housing.csv"
GRAPH_PATH = "static/linear_regression.png"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

df = pd.read_csv(DATA_PATH)
full_df = df.copy()

if "Address" in df.columns:
    used_df = df.drop("Address", axis=1)
else:
    used_df = df.copy()

required_features = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population"
]

target_column = "Price"

missing_columns = [col for col in required_features + [target_column] if col not in used_df.columns]
if missing_columns:
    raise ValueError(f"CSV файлда бул колонкалар жок: {missing_columns}")

X = used_df[required_features]
y = used_df[target_column]

y_pred_all = model.predict(X)

mse_value = mean_squared_error(y, y_pred_all)
r2_value = r2_score(y, y_pred_all)
intercept_value = float(model.intercept_)


def create_regression_graph():
    os.makedirs("static", exist_ok=True)

    plt.figure(figsize=(16, 10))

    plt.scatter(
        y,
        y_pred_all,
        alpha=0.85,
        s=20,
        color="blue",
        label="Actual"
    )

    min_val = min(y.min(), y_pred_all.min())
    max_val = max(y.max(), y_pred_all.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linewidth=3,
        label="Regression Line"
    )

    plt.xlabel("Actual Price ($)", fontsize=16)
    plt.ylabel("Predicted Price ($)", fontsize=16)
    plt.title("Linear Regression Line", fontsize=22)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    plt.grid(True, alpha=0.25)

    plt.tight_layout(pad=2.0)
    plt.savefig(GRAPH_PATH, dpi=300, bbox_inches="tight")
    plt.close()


create_regression_graph()


def safe_float(value):
    try:
        value = str(value).strip().replace(",", "")
        num = float(value)
        if num < 0:
            return None
        return num
    except:
        return None


def format_dataframe_for_html(dataframe, rows=10):
    df_copy = dataframe.head(rows).copy()

    for col in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].map(lambda x: f"{x:,.2f}")

    return df_copy.to_html(
        classes="data-table",
        index=False,
        border=0
    )


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    income = ""
    age = ""
    rooms = ""
    bedrooms = ""
    population = ""

    if request.method == "POST":
        income = request.form.get("income", "").strip()
        age = request.form.get("age", "").strip()
        rooms = request.form.get("rooms", "").strip()
        bedrooms = request.form.get("bedrooms", "").strip()
        population = request.form.get("population", "").strip()

        income_val = safe_float(income)
        age_val = safe_float(age)
        rooms_val = safe_float(rooms)
        bedrooms_val = safe_float(bedrooms)
        population_val = safe_float(population)

        if None in [income_val, age_val, rooms_val, bedrooms_val, population_val]:
            error = "Бардык талааларга 0 же андан чоң сан жазыңыз."
        elif bedrooms_val > rooms_val:
            error = "Bedrooms саны Rooms санынан көп болбошу керек."
        else:
            data = pd.DataFrame([{
                "Avg. Area Income": income_val,
                "Avg. Area House Age": age_val,
                "Avg. Area Number of Rooms": rooms_val,
                "Avg. Area Number of Bedrooms": bedrooms_val,
                "Area Population": population_val
            }])

            prediction_value = model.predict(data)[0]

            if prediction_value < 0:
                prediction_value = 0

            prediction = f"${prediction_value:,.2f}"

    full_table = format_dataframe_for_html(full_df, rows=10)
    used_table = format_dataframe_for_html(used_df, rows=10)

    return render_template(
        "index.html",
        full_table=full_table,
        used_table=used_table,
        prediction=prediction,
        error=error,
        income=income,
        age=age,
        rooms=rooms,
        bedrooms=bedrooms,
        population=population,
        r2_value=round(r2_value, 4),
        mse_value=f"{mse_value:,.2f}",
        intercept_value=f"{intercept_value:,.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)