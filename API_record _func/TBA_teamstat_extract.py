import requests

url = "https://frc-api.firstinspires.org/v3.0/"


payload={}
headers = {
    'Authorization': 'dchen:95038eb2-b003-439d-8134-6006171e3799',
    'If-Modified-Since': ''
}

r = requests.request("GET", url)

print(r.text)