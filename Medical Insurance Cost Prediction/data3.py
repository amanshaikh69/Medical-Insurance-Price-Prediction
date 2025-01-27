import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the insurance dataset
insurance_data = pd.read_csv('insurance.csv')

# Define features and target variable
X = insurance_data[['Age', 'Gender', 'BMI', 'Children', 'Smoker', 'Region']]
y = insurance_data['Charges']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Define a function to predict the insurance premium based on user input
def predict_insurance_premium(age, sex, bmi, children, smoker, region):
    """
    Predicts the insurance premium based on user input.

    Args:
        age (int): The age of the insured.
        sex (str): The sex of the insured (male or female).
        bmi (float): The body mass index of the insured.
        children (int): The number of children of the insured.
        smoker (bool): Whether the insured is a smoker.
        region (str): The region of the insured (northeast, southeast, southwest, northwest).

    Returns:
        float: The predicted insurance premium.
    """

    # Create a DataFrame with the user input
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    