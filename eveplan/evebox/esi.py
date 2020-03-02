import esipy
import requests
import time

# Set up network access
client = esipy.EsiClient(
    transport_adapter = requests.adapters.HTTPAdapter(
        pool_connections=100,
        pool_maxsize=100,
        max_retries=10,
        pool_block=False
    ),
    headers={'User-Agent': 'EVE economy AI playground (a.knieps@fz-juelich.de)'}
)

app = esipy.EsiApp().get_latest_swagger

def check_retry(response):
    if response.status // 100 != 5 and response.status != 429:
        if response.status // 100 != 2:
            raise ValueError('Request returned status {}'.format(response.status))
        
        return False
    
    return True

def request(req, delay = 1):
    while True:
        response = client.request(req)
        
        # If the server didn't fuck it up (5xx) and we didn't exceed the rate limit (429), return result
        if not check_retry(response):
            return response
        
        time.sleep(delay)
        
def multi_request(req, delay = 1):
    responses = [r for _, r in client.multi_request(req)]
    
    while True:
        idx = [i for i in range(len(req)) if check_retry(responses[i])]
        
        if not idx:
            return responses
        
        for i in idx:
            responses[i] = request(req[i])

op = app.op