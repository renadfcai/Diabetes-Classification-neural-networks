#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from adaline import Adaline
from sklearn.linear_model import Perceptron
from madaline import MADALINE
from hebb import HebbianDiabetesClassifier
from sklearn.preprocessing import StandardScaler
from maxnet import Maxnet
import matplotlib.pyplot as plt
import seaborn as sns

# Get the path of the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the folder where the script is located
os.chdir(script_dir)

# Load the dataset
df = pd.read_csv("diabetes.csv")

df["isdiab"] = df["diabetes"].map({"Diabetes": 1, "No diabetes": 0})
df["isdiab"].value_counts()[1]

# print(df.head())

# Removing the unimportatnt features
df.drop(
    [
        "waist",
        "hip",
        "waist_hip_ratio",
    ],
    inplace=True,
    axis=1,
)


# Exploratory data analysis
# sns.lineplot(y ='glucose',x='weight', hue='diabetes', data =df)
# plt.show()

# sns.lineplot(y ='glucose',x='age', hue='gender', data =df)
# plt.show()

# figure, axis = plt.subplots(1,2,figsize=(8,6))
# sns.lineplot(ax=axis[0],x='systolic_bp', y='glucose',hue='diabetes',data=df)
# sns.lineplot(ax=axis[1],x='diastolic_bp', y='glucose',hue='diabetes',data=df)
# plt.show()

# sns.lineplot(x=df.hdl_chol,y= df.glucose,hue=df.diabetes,data=df)
# plt.show()

# sns.countplot(x='diabetes',hue='gender',data=df)
# plt.show()

# sns.lineplot(x=df.cholesterol,y= df.glucose,hue=df.diabetes,data=df)
# plt.show()
# In general, It is seen that higher weight and old Age are two major factor causing diabetes.
# What do we conclude from these graphs?
# 1- Diabetic patients have Higher glucose rate and higher weight as comapred to Non -diabetic ones
# 2- The Age is not directly realted but higher gluose level in oldies can be a cause of Diabetes in them, also the males of age 40 to 80 have higher blood glucose level than females.
# 3- The BP is not directly related to the diabetes, as patients have highest BP are Found to be Non-diabetic.
# 4- Diabetic patients have lower HDL-Cholesterol
# 5- Females being diabetic are more than the males being diabetic.
# 6- Higher cholestrol is seen in the patients having diabetes


# Removing the unimortant features
df1 = df[
    [
        "patient_number",
        "cholesterol",
        "glucose",
        "hdl_chol",
        "age",
        "gender",
        "weight",
        "systolic_bp",
        "diastolic_bp",
        "isdiab",
    ]
]


# Splitting the dataset into X and Y
X = df1[
    [
        "cholesterol",
        "glucose",
        "hdl_chol",
        "age",
        "weight",
        "systolic_bp",
        "diastolic_bp",
    ]
]
y = df["isdiab"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

perceptron = Perceptron(random_state=0)
perceptron.fit(X_train, y_train)
perceptron_predict = perceptron.predict(X_test)
# Perceptron Accuracy 91.6%
# print("perceptron accuracy : ", accuracy_score(y_test, perceptron_predict))


def predict_diabetes_Perceptron(
    cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp
):
    user_inputs = np.array(
        [
            cholesterol,
            glucose,
            hdl_chol,
            age,
            float(weight) * 2.2,
            systolic_bp,
            diastolic_bp,
        ]
    ).reshape(1, -1)
    user_df = pd.DataFrame(
        user_inputs,
        columns=[
            "cholesterol",
            "glucose",
            "hdl_chol",
            "age",
            "weight",
            "systolic_bp",
            "diastolic_bp",
        ],
    )

    prediction = perceptron.predict(user_df)

    return prediction[0]


mlp = MLPClassifier(solver="lbfgs", activation="identity", random_state=1)
mlp.fit(X_train, y_train)
mlp_predict = mlp.predict(X_test)
# # MLP Accuracy 92.9%
# print("mlp accuracy : ", accuracy_score(y_test, mlp_predict))

# conf_matrix_mlp = confusion_matrix(y_test, mlp_predict)
# print("Confusion matrix for MLP model:")
# print(conf_matrix_mlp)

# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix_mlp, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.title("Confusion Matrix - MLP Model")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()


def predict_diabetes_MLP(
    cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp
):
    user_inputs = np.array(
        [
            cholesterol,
            glucose,
            hdl_chol,
            age,
            float(weight) * 2.2,
            systolic_bp,
            diastolic_bp,
        ]
    ).reshape(1, -1)
    user_df = pd.DataFrame(
        user_inputs,
        columns=[
            "cholesterol",
            "glucose",
            "hdl_chol",
            "age",
            "weight",
            "systolic_bp",
            "diastolic_bp",
        ],
    )

    prediction = mlp.predict(user_df)

    return prediction[0]


adaline = Adaline()
adaline.fit(X_train, y_train)
adaline_predict = adaline.predict(X_test)


# # ADALINE Accuracy 83.97%
# print("ADALINE accuracy : ", accuracy_score(adaline_predict, y_test))
# conf_matrix_adaline = confusion_matrix(y_test, adaline_predict)
# print("Confusion matrix for ADALINE model:")
# print(conf_matrix_adaline)

# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix_adaline, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.title("Confusion Matrix - ADALINE Model")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()


def predict_diabetes_Adaline(
    cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp
):
    # Convert input variables to float
    cholesterol = float(cholesterol)
    glucose = float(glucose)
    hdl_chol = float(hdl_chol)
    age = float(age)
    weight = float(weight) * 2.2  # Convert weight to pounds
    systolic_bp = float(systolic_bp)
    diastolic_bp = float(diastolic_bp)

    # Create NumPy array and DataFrame from input variables
    user_inputs = np.array(
        [cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp]
    ).reshape(1, -1)
    user_df = pd.DataFrame(
        user_inputs,
        columns=[
            "cholesterol",
            "glucose",
            "hdl_chol",
            "age",
            "weight",
            "systolic_bp",
            "diastolic_bp",
        ],
    )

    # Make prediction using Adaline model
    prediction = adaline.predict(user_df)

    return prediction[0]


# Standardize the features for Maxnet learning
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate Maxnet model
maxnet_model = Maxnet()

# Train the model
maxnet_model.train(X_train_scaled)

# Predict on test set
y_pred = maxnet_model.predict(X_test_scaled)
# # Measure accuracy
# accuracy = accuracy_score(y_test, y_pred)
# # maxnet accuracy is 89.1%
# print("maxnet accuracy:", accuracy)

# conf_matrix_maxnet = confusion_matrix(y_test, y_pred)
# print("Confusion matrix for Maxnet model:")
# print(conf_matrix_maxnet)

# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix_maxnet, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.title("Confusion Matrix - Maxnet Model")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()


def predict_diabetes_maxnet(
    cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp
):
    # Convert input variables to float
    cholesterol = float(cholesterol)
    glucose = float(glucose)
    hdl_chol = float(hdl_chol)
    age = float(age)
    weight = float(weight) * 2.2  # Convert weight to pounds
    systolic_bp = float(systolic_bp)
    diastolic_bp = float(diastolic_bp)

    # Create NumPy array and DataFrame from input variables
    user_inputs = np.array(
        [cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp]
    ).reshape(1, -1)
    user_df = pd.DataFrame(
        user_inputs,
        columns=[
            "cholesterol",
            "glucose",
            "hdl_chol",
            "age",
            "weight",
            "systolic_bp",
            "diastolic_bp",
        ],
    )

    # Make prediction using Adaline model
    prediction = maxnet_model.predict(user_df)

    return prediction[0]


# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

num_units = 3  # Number of Adaline-like units in the MADALINE
input_dim = 10  # Dimensionality of input features

# Create MADALINE model
model = MADALINE(num_units)

# Compile the model
model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Make predictions
madaline_predict = model.predict(X_test)
madaline_predict_binary = np.round(madaline_predict)


def predict_diabetes_Madaline(
    cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp
):
    # Convert input variables to float
    cholesterol = float(cholesterol)
    glucose = float(glucose)
    hdl_chol = float(hdl_chol)
    age = float(age)
    weight = float(weight) * 2.2  # Convert weight to pounds
    systolic_bp = float(systolic_bp)
    diastolic_bp = float(diastolic_bp)

    # Create NumPy array and DataFrame from input variables
    user_inputs = np.array(
        [cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp]
    ).reshape(1, -1)
    user_df = pd.DataFrame(
        user_inputs,
        columns=[
            "cholesterol",
            "glucose",
            "hdl_chol",
            "age",
            "weight",
            "systolic_bp",
            "diastolic_bp",
        ],
    )
    user_inputs = np.array(user_df)
    prediction = model.predict(user_inputs)
    return prediction[0]


# Calculate confusion matrix
# conf_matrix_madaline = confusion_matrix(y_test, madaline_predict_binary)
# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix_madaline, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.title("Confusion Matrix - MADALINE Model")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()


# Standardize the features for Hebbian learning
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensure labels are -1 or +1
y_train_encoded = np.where(y_train > 0, 1, -1)
y_test_encoded = np.where(y_test > 0, 1, -1)

hebbian = HebbianDiabetesClassifier()
hebbian.fit(X_train_scaled, y_train_encoded)

hebbian_predict = hebbian.predict(X_test_scaled)
hebbian_accuracy = accuracy_score(y_test_encoded, hebbian_predict)
# accuracy hebbian : 92.3%
# print("Hebbian accuracy:", hebbian_accuracy)


def predict_diabetes_Hebbian(
    cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp
):
    # Convert input variables to float
    cholesterol = float(cholesterol)
    glucose = float(glucose)
    hdl_chol = float(hdl_chol)
    age = float(age)
    weight = float(weight) * 2.2  # Convert weight to pounds
    systolic_bp = float(systolic_bp)
    diastolic_bp = float(diastolic_bp)

    # Create NumPy array and DataFrame from input variables
    user_inputs = np.array(
        [cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp]
    ).reshape(1, -1)
    user_df = pd.DataFrame(
        user_inputs,
        columns=[
            "cholesterol",
            "glucose",
            "hdl_chol",
            "age",
            "weight",
            "systolic_bp",
            "diastolic_bp",
        ],
    )

    user_inputs = scaler.transform(user_df)
    prediction = hebbian.predict(user_inputs)
    return prediction[0]


# conf_matrix_hebbian = confusion_matrix(y_test_encoded, hebbian_predict)
# print("Confusion matrix for Hebbian model:")
# print(conf_matrix_hebbian)

# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix_hebbian, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.title("Confusion Matrix - Hebbian Model")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()
