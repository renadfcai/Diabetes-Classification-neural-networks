from flask import Flask, request, render_template
import predictDiabetes
import threading

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # getting patient info from Form
    cholesterol_value = request.form["cholesterol"]
    glucose_value = request.form["glucose"]
    hdl_chol_value = request.form["hdl_chol"]
    age_value = request.form["age"]
    weight_value = request.form["weight"]
    systolic_bp_value = request.form["systolic_bp"]
    diastolic_bp_value = request.form["diastolic_bp"]

    def run_perceptron():
        result = predictDiabetes.predict_diabetes_Perceptron(
            cholesterol_value,
            glucose_value,
            hdl_chol_value,
            age_value,
            weight_value,
            systolic_bp_value,
            diastolic_bp_value,
        )
        return result

    def run_mlp():
        result = predictDiabetes.predict_diabetes_MLP(
            cholesterol_value,
            glucose_value,
            hdl_chol_value,
            age_value,
            weight_value,
            systolic_bp_value,
            diastolic_bp_value,
        )
        return result

    def run_adaline():
        result = predictDiabetes.predict_diabetes_Adaline(
            cholesterol_value,
            glucose_value,
            hdl_chol_value,
            age_value,
            weight_value,
            systolic_bp_value,
            diastolic_bp_value,
        )
        return result

    def run_hebb():
        result = predictDiabetes.predict_diabetes_Hebbian(
            cholesterol_value,
            glucose_value,
            hdl_chol_value,
            age_value,
            weight_value,
            systolic_bp_value,
            diastolic_bp_value,
        )
        return result

    def run_madaline():
        result = predictDiabetes.predict_diabetes_Madaline(
            cholesterol_value,
            glucose_value,
            hdl_chol_value,
            age_value,
            weight_value,
            systolic_bp_value,
            diastolic_bp_value,
        )
        return result

    def run_maxnet():
        result = predictDiabetes.predict_diabetes_maxnet(
            cholesterol_value,
            glucose_value,
            hdl_chol_value,
            age_value,
            weight_value,
            systolic_bp_value,
            diastolic_bp_value,
        )
        return result

    return render_template(
        "result.html",
        predictions={
            "perceptron_prediction": run_perceptron(),
            "mlp_prediction": run_mlp(),
            "adaline_prediction": run_adaline(),
            "hebbian_prediction": run_hebb(),
            "madaline_prediction": run_madaline(),
            "maxnet_prediction": run_maxnet(),
        },
    )


if __name__ == "__main__":
    app.run(debug=False)

# Updateing acurecy 