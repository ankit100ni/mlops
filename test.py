import joblib

# Load the trained model from the .pkl file
with open('trained_model.pkl', 'rb') as model_file:
    clf = joblib.load(model_file)

# Example input data (replace with your actual input data)
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Example input for Iris dataset

# Make predictions using the loaded model
predictions = clf.predict(new_data)

# Display the predictions
print("Predictions:", predictions)
