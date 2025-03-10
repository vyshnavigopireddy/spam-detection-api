import requests

# Define API endpoint
url = "http://127.0.0.1:5000/predict"

# Sample message for testing
test_data = {"message": "Congratulations! You've won a free lottery ticket. Claim now!"}

# Send POST request
response = requests.post(url, json=test_data)

# Print the response
print("Response:", response.json())
