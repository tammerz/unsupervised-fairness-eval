import urllib.request
import re

url = 'https://upload.wikimedia.org/wikipedia/commons/9/92/Zegel_Vrije_Universiteit_Amsterdam.svg'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
response = urllib.request.urlopen(req)
svg_data = response.read().decode('utf-8')

# Force white fill
svg_data = re.sub(r'fill="[^"]+"', 'fill="#ffffff"', svg_data)
# Sometime the fill is in style attributes
svg_data = re.sub(r'fill:[^;"]+', 'fill:#ffffff', svg_data)

with open('vu_logo.svg', 'w', encoding='utf-8') as f:
    f.write(svg_data)
print("Saved vu_logo.svg")
