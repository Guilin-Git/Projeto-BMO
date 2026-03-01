import json

with open('applio_api.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('parsed_api_utf8.txt', 'w', encoding='utf-8') as out:
    for k, v in list(data.get('unnamed_endpoints', {}).items()) + list(data.get('named_endpoints', {}).items()):
        params = v.get('parameters', [])
        if len(params) > 50:
            out.write(f"=== Endpoint {k} ===\n")
            out.write(f"Total Params: {len(params)}\n")
            for i, p in enumerate(params):
                out.write(f"  [{i}]: {p.get('parameter_name')} ({p.get('component')})\n")
