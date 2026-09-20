import urllib.request, re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'tr-TR,tr;q=0.9',
}

url = 'https://www.tff.org/default.aspx?pageID=198'
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=20) as resp:
    html = resp.read().decode('utf-8', errors='replace')

rows = re.findall(r'<tr>(.*?)</tr>', html, re.DOTALL)
count = 0
for r in rows:
    if 'macId=' in r and re.search(r'(\d+)\s*-\s*(\d+)', r):
        macids = re.findall(r'macId=(\d+)', r)
        names = re.findall(r'<a[^>]*>([^<]+)</a>', r)
        scores = re.findall(r'(\d+)\s*-\s*(\d+)', r)
        print(f'macId: {macids}, isimler: {names[:5]}, skor: {scores}')
        count += 1
        if count >= 5:
            break
