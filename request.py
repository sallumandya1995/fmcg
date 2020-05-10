import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'SLSMAN_CD':61,'PROD_CD':60,'PLAN_MONTH':6,'TARGET_IN_EA':750})

print(r.json())
