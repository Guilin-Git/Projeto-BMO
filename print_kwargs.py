import json
# Load from API export
data = json.load(open('applio_api.json', encoding='utf-8'))
params = data['named_endpoints']['/enforce_terms']['parameters']
print(f"kwargs = {{")
for p in params:
    name = p['parameter_name']
    default = p.get('parameter_default')
    # Use proper Python literals for defaults
    if isinstance(default, str):
        default_str = f"'{default}'"
    elif isinstance(default, bool):
        default_str = str(default)
    elif default is None:
        default_str = "None"
    else:
        default_str = str(default)
    print(f"    '{name}': {default_str},")
print(f"}}")
