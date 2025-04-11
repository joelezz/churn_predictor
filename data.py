from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd


def calc_data():
    # Load customer data
    customer_data = pd.read_csv('customer_data.csv')
    # Display the first few rows of the dataset
    print(customer_data.head())

    # Fill missing values and encode categorical columns
    tenure_mean = customer_data['tenure'].mean()


    state_encoder = LabelEncoder()
    customer_data['encoded_state'] = state_encoder.fit_transform(customer_data['state'])
    # Scale numerical columns
    scaler = StandardScaler()
    customer_data[['tenure', 'monthly_charges']] = scaler.fit_transform(customer_data[['tenure', 'monthly_charges']])

    print(customer_data.head())

    # Split the data into features and target
    x = customer_data[['tenure', 'monthly_charges', 'encoded_state']]
    y = customer_data['churn']

    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Make predictions
    y_pred = model.predict(x_test)

    # Evaluate the model
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

    # Calculate average churn probability
    churn_probabilities = model.predict_proba(x_test)[:, 1]
    avg_churn_prob = churn_probabilities.mean()
    # Identify high-risk customers (e.g., those with a churn probability above a threshold)
    high_risk_customers = x_test[churn_probabilities > avg_churn_prob]

    # Create lists to store churn rates and high-risk counts by state
    churn_rate_by_state = []
    high_risk_by_state = []

    # Group by state and calculate churn rate and high-risk count
    for state, group in customer_data.groupby('state'):
        # Calculate the churn for the state
        churn_rate, high_risk_count = calculate_churn_and_high_risk(group['churn'], avg_churn_prob)
        # Append the state and churn rate to the list
        churn_rate_by_state.append({'state': state, 'churn_rate': churn_rate})
        # Append the state and high risk count to the list
        high_risk_by_state.append({'state': state, 'high_risk': high_risk_count})

    # Convert lists to DataFrames
    churn_rate_by_state_df = pd.DataFrame(churn_rate_by_state)
    high_risk_by_state_df = pd.DataFrame(high_risk_by_state)

    # Store the results in a text file for autograding. Do not modify this code.
    with open('churn_results.txt', 'w') as f:
        f.write("Do not modify this file. It is used for autograding the processed data from the lab.\n\n")
        f.write(f"Average Churn Probability: {avg_churn_prob}\n\n")
        f.write(f"High-Risk Customers: {high_risk_customers}\n\n")
        f.write(f"Churn Rate by State:\n {churn_rate_by_state_df}\n\n")
        f.write(f"High-Risk Customers by State:\n {high_risk_by_state_df}")

    return avg_churn_prob, high_risk_customers, churn_rate_by_state_df, high_risk_by_state_df

def calculate_churn_and_high_risk(churn_probabilities, avg_churn_prob):
    """Calculate churn rate and count high-risk customers."""
    churn_rate = (churn_probabilities >= avg_churn_prob).mean()  # mean of churn probabilities above the threshold
    high_risk_count = (churn_probabilities > avg_churn_prob).sum()  # count high-risk customers

    return churn_rate, high_risk_count

if __name__ == "__main__":
    calc_data()