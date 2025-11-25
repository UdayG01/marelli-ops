#!/usr/bin/env python3
"""
Simple test script for bidirectional Node-RED ↔ Django communication
"""

import requests
import json
import time

# Configuration
DJANGO_BASE_URL = "http://localhost:8000/api/ml"
NODE_RED_BASE_URL = "http://localhost:1880"

def test_django_to_node_red():
    """Test Django → Node-RED communication (existing functionality)"""
    print("🔄 Testing Django → Node-RED...")
    
    try:
        response = requests.post(f"{DJANGO_BASE_URL}/test-node-red/")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Django → Node-RED: {data['message']}")
            return True
        else:
            print(f"❌ Django → Node-RED failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Django → Node-RED error: {e}")
        return False

def test_node_red_to_django():
    """Test Node-RED → Django communication (new functionality)"""
    print("🔄 Testing Node-RED → Django...")
    
    test_messages = [
        {"id": "hello", "message": "Hello from test script!"},
        {"id": "test123", "message": "This is a test message"},
        {"id": "trigger_request", "message": "Please trigger camera"}
    ]
    
    success_count = 0
    
    for msg in test_messages:
        try:
            response = requests.post(
                f"{DJANGO_BASE_URL}/from-node-red/",
                json=msg,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Message '{msg['id']}': {data['reply']}")
                success_count += 1
            else:
                print(f"❌ Message '{msg['id']}' failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Message '{msg['id']}' error: {e}")
    
    return success_count == len(test_messages)

def test_bidirectional_communication():
    """Test both directions"""
    print("🚀 Starting bidirectional communication test...\n")
    
    # Test Django → Node-RED
    django_to_node_red_ok = test_django_to_node_red()
    time.sleep(1)
    
    # Test Node-RED → Django  
    node_red_to_django_ok = test_node_red_to_django()
    
    print("\n📊 Test Results:")
    print(f"Django → Node-RED: {'✅ PASS' if django_to_node_red_ok else '❌ FAIL'}")
    print(f"Node-RED → Django: {'✅ PASS' if node_red_to_django_ok else '❌ FAIL'}")
    
    if django_to_node_red_ok and node_red_to_django_ok:
        print("\n🎉 Bidirectional communication is working!")
    else:
        print("\n⚠️  Some tests failed. Check your setup.")

if __name__ == "__main__":
    test_bidirectional_communication()
