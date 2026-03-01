import json

with open('applio_api.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for k, v in list(data.get('unnamed_endpoints', {}).items()) + list(data.get('named_endpoints', {}).items()):
    params = v.get('parameters', [])
    if len(params) > 50:
        print(f"=== Endpoint {k} ===")
        print(f"Total Params: {len(params)}")
        for i, p in enumerate(params):
            print(f"  [{i}]: {p.get('parameter_name')} ({p.get('component')})")
