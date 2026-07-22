# Salary Prediction using Polynomial Regression

**Name:** ARAVAPALLI M ADITYA SRIRAM\
**Registration Number:** 23MIM10130\
**Application Number:** IN26010416\
**Batch Number:** 1A  \
**Email:** aravapalli.23mim10130@vitbhopal.ac.in

## Objective
The objective of this assignment is to develop a Polynomial Regression model to estimate employee salaries based on their position level, addressing the inherently non-linear relationship between corporate hierarchy and compensation.

## Dataset Link
The dataset used is the **Position Salaries Dataset**.  
Following submission guidelines, the dataset is not hosted in this repository. You can download it directly from Kaggle:  
[Position Salaries Dataset on Kaggle](https://www.kaggle.com/datasets/akram24/position-salaries)

## Libraries Used
* **Pandas:** Data loading and inspection.
* **NumPy:** Mathematical operations and grid generation for curve plotting.
* **Matplotlib:** Creating scatter plots and the continuous polynomial regression curve.
* **Scikit-learn:** Train-test splitting, generating polynomial features (degree 3), model training, and calculating evaluation metrics (MAE, MSE, R-squared).

## Methodology
1. **Data Understanding:** Analyzed the dataset to identify `Level` as the input feature and `Salary` as the target variable. Examined summary statistics.
2. **Data Preprocessing:** Verified the absence of missing values. Selected the appropriate numerical feature array and split the data into 80% training and 20% testing sets.
3. **Model Development:** Transformed the input feature using `PolynomialFeatures` with a degree of 3. Trained a Linear Regression model on this transformed data to create a polynomial model, then generated predictions for the test set.
4. **Model Evaluation:** Calculated MAE, MSE, and the R-squared score. Visualized the model's accuracy by plotting the continuous polynomial curve against a scatter plot of the original data.

## Results
* **Mean Absolute Error (MAE):** 70635.25
* **Mean Squared Error (MSE):** 6263853282.86
* **R-squared Score:** 0.8763

## Conclusion
Polynomial Regression was effective for this dataset because salaries rise non-linearly with position level. The cubic model captured the accelerating growth pattern at higher levels and produced strong predictive performance with low error values on the test set. Compared with Linear Regression, polynomial regression can bend to follow curved patterns rather than forcing a straight line through every point, which makes it more suitable for salary structures that increase sharply at senior and executive ranks. Another advantage of polynomial regression is that it can model these high-level jumps more realistically, improving the usefulness of predictions for managerial and leadership roles. Overall, this approach provides a better fit for the non-linear relationship between position level and salary in this assignment.
