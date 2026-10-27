# src/alerting.py
import requests
import json

def send_discord_alert(webhook_url: str, title: str, content: str) -> tuple:
    """
    Sends a formatted alert to a Discord webhook.

    Args:
        webhook_url (str): The Discord webhook URL.
        title (str): The title of the alert embed.
        content (str): The main content/description of the alert.

    Returns:
        tuple: The status code and response text from the request.
    """
    if not webhook_url:
        return 400, "Webhook URL is missing."

    # Discord webhook expects a specific JSON payload format
    payload = {
        "embeds": [{
            "title": title,
            "description": content,
            "color": 15158332  # A red color
        }]
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
        return response.status_code, response.text
    except requests.exceptions.RequestException as e:
        return 500, str(e)