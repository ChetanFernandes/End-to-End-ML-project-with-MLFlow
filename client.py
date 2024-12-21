import requests

# Define the payload
#url = 'http://127.0.0.1:8080/train'

payload =  {"stage": ["data_ingestion","data_transformation","model_training"]}

try:
    # Send a POST request to the Flask endpoint
    response = requests.post(url, json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        print("Pipeline Response:", response.json())
    else:
        print(f"Failed to trigger pipeline: {response.status_code}, {response.text}")
    

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    
