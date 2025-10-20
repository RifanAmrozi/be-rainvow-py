import requests

class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def post_stats(self, payload: dict) -> int:
        r = requests.post(self.base_url, json=payload)
        return r.status_code
