"""Free Proxy Havuzu - calisan HTTP/SOCKS proxy'leri toplar ve flashscore icin test eder.

Kaynaklar (ustel github raw listeleri):
  - TheSpeedX/PROXY-List        (http, socks4, socks5)
  - monosans/proxy-list         (http, socks4, socks5)
  - jetkai/proxy-list           (online-proxies)
  - mertguvencli/http-proxy-list
  - clarketm/proxy-list         (raw, "ip:port country" format)

Surec:
  1. Ham listeleri cek (yukaridakiler) -> ip:port adaylari.
  2. Hedefe (flashscore.com.tr) HTTP GET ile test -> calisanlari sec.
  3. `tahminler/free_proxies.json` cache; TTL 10dk (proxy'ler cok sayilir yuk.

Not: Flashscore'a erisim TR'den acik; proxy'ler IP bazli blok/429 durumlarinda
devreye girer. cagri yerine gorunmeyen timeout'lar icin `timeout` parametresi.
"""
import io
import json
import os
import random
import re
import time
import urllib.request

_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tahminler", "free_proxies.json")
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
_PROXY_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}:\d{2,5}\b")

_SOURCES = [
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks4.txt",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks4.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks5.txt",
    "https://raw.githubusercontent.com/mertguvencli/http-proxy-list/main/proxy-list/data.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-http.txt",
    "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-socks5.txt",
]

_TARGET = "https://www.flashscore.com.tr/"


def _valid_ip(ip_port: str) -> bool:
    m = _PROXY_RE.match(ip_port.strip())
    if not m:
        return False
    try:
        ip, port = ip_port.split(":")
        port = int(port)
        if not (0 < port < 65536):
            return False
        parts = [int(x) for x in ip.split(".")]
        return all(0 <= x <= 255 for x in parts) and len(parts) == 4
    except Exception:
        return False


def collect_candidates() -> list:
    """Kaynaklardan ham ip:port adaylarini toplar (tekrarlari tekil)."""
    seen = set()
    for url in _SOURCES:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=20) as resp:
                text = resp.read().decode("utf-8", "replace")
            for token in text.splitlines():
                token = token.strip()
                if not token:
                    continue
                # clarketm: "ip:port  country  ..." -> ilk token
                cand = token.split()[0]
                if _valid_ip(cand):
                    seen.add(cand)
        except Exception:
            continue
    lst = list(seen)
    random.shuffle(lst)
    return lst


def _test_proxy(proxy: str, timeout: int = 6) -> bool:
    """Hedefe proxy uzerinden erisimi dener (HTTP 200 + gecersiz proxy?)."""
    try:
        handler = urllib.request.ProxyHandler({"http": "http://" + proxy, "https": "http://" + proxy})
        opener = urllib.request.build_opener(handler)
        req = urllib.request.Request(_TARGET, headers={"User-Agent": _UA})
        with opener.open(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _load_cache():
    try:
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"fetched_at": 0, "proxies": [], "tested": 0}


def _save_cache(data: dict):
    os.makedirs(os.path.dirname(_JSON_PATH), exist_ok=True)
    tmp = _JSON_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, _JSON_PATH)


def get_working_proxies(limit: int = 20, cache_ttl: int = 600) -> list:
    """Calisan (flashscore'a ulasabilen) proxy listesini dondurur.

    Cache'te gecerli veri varsa onu kullanir; yoksa topla ve test et.
    Verimlilik: ilk `limit` ayan adayi test eder (her proxy ~timeout sn surer).
    """
    cache = _load_cache()
    if time.time() - cache.get("fetched_at", 0) < cache_ttl and cache.get("proxies"):
        return cache["proxies"]

    candidates = collect_candidates()
    working = []
    for pytest_cand in candidates[: max(limit * 3, 30)]:
        if len(working) >= limit:
            break
        if _test_proxy(pytest_cand):
            working.append(pytest_cand)

    _save_cache({"fetched_at": time.time(), "proxies": working, "tested": len(working)})
    return working


def get_proxy(limit: int = 20, cache_ttl: int = 600) -> str | None:
    proxies = get_working_proxies(limit=limit, cache_ttl=cache_ttl)
    return random.choice(proxies) if proxies else None


if __name__ == "__main__":
    sys_stdout_saved = getattr(__import__("sys"), "stdout")
    import sys
    sys.stdout = io.TextIOWrapper(sys_stdout_saved.buffer, encoding="utf-8", errors="replace")
    ok = get_working_proxies(limit=20)
    print(f"calisan proxy: {len(ok)}")
    for p in ok[:20]:
        print("  ", p)