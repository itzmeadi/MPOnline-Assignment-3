import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# Function to Load Dataset
# =====================================================
def load_dataset():
    folder = Path(__file__).parent
    file = folder / "Position_Salaries.csv"

    if file.exists():
        data = pd.read_csv(file)
    else:
        print("Dataset not found! Creating sample dataset...")

        data = pd.DataFrame({
            "Position": [
                "Business Analyst",
                "Junior Consultant",
                "Senior Consultant",
                "Manager",
                "Country Manager",
                "Region Manager",
                "Partner",
                "Senior Partner",
                "C-Level",
                "CEO"
            ],
            "Level": [1,2,3,4,5,6,7,8,9,10],
            "Salary":[45000,50000,60000,80000,110000,
                      150000,200000,300000,500000,1000000]
        })

        data.to_csv(file, index=False)

    return data, folder


# =====================================================
# Display Dataset Details
# =====================================================
def explore_dataset(data):

    print("\n========== TASK 1 : DATA UNDERSTANDING ==========\n")

    print("First Five Records:\n")
    print(data.head())

    print("\nDataset Information:\n")
    data.info()

    print("\nSummary Statistics:\n")
    print(data.describe())

    print("\nMissing Values:\n")
    print(data.isnull().sum())


# =====================================================
# Train Polynomial Regression
# =====================================================
def build_model(data):

    X = data[["Level"]]
    y = data["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=0
    )

    poly = PolynomialFeatures(degree=4)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()

    model.fit(X_train_poly, y_train)

    prediction = model.predict(X_test_poly)

    print("\n========== TASK 2 : MODEL EVALUATION ==========\n")

    print("Mean Absolute Error :", round(mean_absolute_error(y_test, prediction),2))

    print("Mean Squared Error :", round(mean_squared_error(y_test, prediction),2))

    print("R2 Score :", round(r2_score(y_test, prediction),4))

    return model, poly


# =====================================================
# Plot Graph
# =====================================================
def draw_graph(data, model, poly, location):

    x = data[["Level"]]
    y = data["Salary"]

    smooth = np.linspace(
        x.min().values[0],
        x.max().values[0],
        200
    ).reshape(-1,1)

    plt.figure(figsize=(8,6))

    plt.scatter(
        x,
        y,
        color="red",
        s=60,
        label="Actual Data"
    )

    plt.plot(
        smooth,
        model.predict(poly.transform(smooth)),
        color="blue",
        linewidth=2,
        label="Polynomial Curve"
    )

    plt.title("Polynomial Regression")

    plt.xlabel("Position Level")

    plt.ylabel("Salary")

    plt.legend()

    plt.grid(True)

    plt.savefig(location/"Polynomial_Regression_Output.png")

    plt.close()


# =====================================================
# Main Program
# =====================================================
def main():

    dataset, path = load_dataset()

    explore_dataset(dataset)

    model, poly = build_model(dataset)

    draw_graph(dataset, model, poly, path)

    print("\nGraph saved successfully.")

    print("\nObservations")

    print("1. Salary increases non-linearly with employee level.")

    print("2. Polynomial Regression models the salary trend effectively.")

    print("3. Higher-level positions experience a rapid increase in salary.")

    print("4. Degree-4 Polynomial provides a better curve than simple Linear Regression.")


if __name__ == "__main__":
    main()