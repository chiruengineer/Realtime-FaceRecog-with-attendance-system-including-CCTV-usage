import json
import os
import cv2
import urllib.request

def test_camera_connection(source):
    """
    Test if a camera connection can be established
    """
    try:
        if isinstance(source, str):
            if source.startswith('rtsp://') or source.startswith('http://'):
                # For IP cameras, try to open a connection
                stream = urllib.request.urlopen(source)
                return True
        else:
            # For webcams, try to open the device
            cap = cv2.VideoCapture(source)
            if cap is None or not cap.isOpened():
                return False
            cap.release()
            return True
    except Exception as e:
        print(f"Error testing camera connection: {str(e)}")
        return False

def get_camera_config():
    """
    Get current camera configuration
    """
    config_file = 'camera_config.json'
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return None

def update_camera_config(ip_camera_url=None, resolution=None, fps=None):
    """
    Update camera configuration
    """
    config_file = 'camera_config.json'
    
    # Load existing config or create new
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "ip_camera_url": "",
            "camera_settings": {
                "resolution": {
                    "width": 1280,
                    "height": 720
                },
                "fps": 30
            }
        }
    
    # Update values
    if ip_camera_url is not None:
        # Test the camera connection before saving
        if ip_camera_url and not test_camera_connection(ip_camera_url):
            raise Exception("Could not connect to the camera with the provided settings")
        config['ip_camera_url'] = ip_camera_url
    
    if resolution is not None:
        config['camera_settings']['resolution'] = resolution
    
    if fps is not None:
        config['camera_settings']['fps'] = fps
    
    # Save config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

def get_available_camera():
    """
    Get the first available camera
    """
    # First try IP camera from config
    config = get_camera_config()
    if config and config.get('ip_camera_url'):
        if test_camera_connection(config['ip_camera_url']):
            return config['ip_camera_url']
    
    # Then try webcams
    for i in range(10):
        if test_camera_connection(i):
            return i
    
    return None

