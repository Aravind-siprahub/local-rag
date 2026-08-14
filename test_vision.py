import requests
import io
import sys
import os
from PIL import Image
from dotenv import load_dotenv

# Explicitly load the backend .env file so settings load properly
load_dotenv(os.path.join(os.path.dirname(__file__), 'backend', '.env'))

# Add backend directory to sys.path to import local-rag modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))
from app.api.security import create_access_token


def test_image_analysis():
    print("Testing image analysis API...")
    
    # Generate a valid access token for the demo user
    # Using the demo user ID from your logs
    user_id = "09e6e22a-36cf-421f-a0f2-8c7950f09a39"
    token = create_access_token(user_id)
    
    # Create a dummy image for testing
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    # API endpoint (adjust if your backend runs on a different port/host)
    url = "http://localhost:8000/api/chat"
    
    # We use multipart/form-data as per chat.py
    files = {
        'file': ('test.png', img_bytes, 'image/png')
    }
    data = {
        'question': 'What color is this image?'
    }
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            resp_json = response.json()
            print("Response:", resp_json.get("answer"))
        else:
            print("Error:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("Failed to connect. Make sure the backend server (python run.py) is running on http://localhost:8000")

if __name__ == "__main__":
    test_image_analysis()
