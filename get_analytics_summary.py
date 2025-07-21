import requests
import json

def get_analytics_summary():
    """
    Fetches the analytics summary from the local API
    and prints the JSON response, pretty-printed.
    """
    url = "http://localhost:8005/analytics/summary"
    print(f"Fetching data from {url}...")
    
    try:
        response = requests.get(url)
        # Raise an HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()
        
        # Convert the Python dictionary to a well-formatted JSON string
        # The indent=2 argument pretty-prints the JSON, similar to ConvertTo-Json
        formatted_json = json.dumps(data, indent=2)
        
        print("\n--- Analytics Summary ---")
        print(formatted_json)
        
    except requests.exceptions.ConnectionError as e:
        print(f"\nError: Could not connect to the server at {url}.")
        print("Please ensure the Docker containers are running and the port is correct.")
    except requests.exceptions.HTTPError as e:
        print(f"\nHTTP Error: {e.response.status_code} {e.response.reason}")
        print(f"Response content: {e.response.text}")
    except json.JSONDecodeError:
        print("\nError: Failed to decode JSON from the response.")
        print(f"Received content: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    get_analytics_summary()
