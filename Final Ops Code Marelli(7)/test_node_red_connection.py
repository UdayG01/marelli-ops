import requests
import json

# Test Node-RED connection
def test_node_red():
    url = "http://localhost:1880/inspection-result"
    data = {
        "image_id": "TEST_MANUAL",
        "overall_result": "PASS",
        "source": "manual_test",
        "capture_type": "manual",
        "timestamp": "2025-01-26T12:00:00"
    }
    
    try:
        response = requests.post(url, json=data, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        print("✅ Success! Check Node-RED debug panel.")
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Node-RED is not running or not accessible")
    except requests.exceptions.Timeout:
        print("❌ Timeout: Node-RED didn't respond within 5 seconds")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_node_red()
