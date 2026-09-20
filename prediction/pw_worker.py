"""Tek Playwright isci thread'i - greenlet thread hatalarini onler.

Playwright sync API'si her nesnesini olusturan greenlet/thread'e baglar:
'page' veya 'browser' baska bir thread'ten kullanilirsa
    greenlet.error: Cannot switch to a different thread
hatasi cikar. Flask'te request thread'leri + arka plan "Canli kontrol"
thread'i ayni anda Playwright cagirinca bu ortaya cikar.

Bu modul TUM Playwright islerini tek daemon thread uzerinde calistirir ve o
thread'te TEK bir playwright + browser tutar (ayni thread'te ikinci bir
sync_playwright().start() cagrisi greenlet karisikligi yaratir, o yuzden ayni
browser yeniden kullanilir).

Kullanim:
    from prediction.pw_worker import run_on_page, run_with_browser

    # Ortak page uzerinde calisma (sofascore API fetch'leri gibi):
    out = run_on_page(lambda page: page.evaluate("..."), timeout=180)

    # Kendi context'ini ortak browser'dan acan uzun isler:
    out = run_with_browser(lambda browser, page: ... , timeout=240)
"""
import queue
import threading

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


class _Worker:
    def __init__(self):
        self._q = queue.Queue()
        self._pw = None
        self._browser = None
        self._page = None
        self._thread = threading.Thread(
            target=self._run, name="playwright-worker", daemon=True
        )
        self._thread.start()

    def _ensure(self):
        if self._browser is not None and not self._browser.is_connected():
            self._browser = None
        if self._browser is None:
            from playwright.sync_api import sync_playwright

            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=True)
            self._page = None
        if self._page is None or self._page.is_closed():
            ctx = self._browser.new_context(
                user_agent=_UA,
                locale="tr-TR",
                viewport={"width": 1280, "height": 900},
            )
            self._page = ctx.new_page()
        return self._browser, self._page

    def _run(self):
        while True:
            kind, cb, fut, args, kwargs = self._q.get()
            try:
                if kind == "on_page":
                    _b, page = self._ensure()
                    res = cb(page, *args, **kwargs)
                elif kind == "with_browser":
                    browser, page = self._ensure()
                    res = cb(browser, page, *args, **kwargs)
                else:
                    res = cb(*args, **kwargs)
                fut.set_result(res)
            except Exception as exc:  # noqa: BLE001
                fut.set_exception(exc)

    def _submit(self, kind, cb, args, kwargs, timeout):
        fut = __import__("concurrent.futures").futures.Future()
        self._q.put((kind, cb, fut, args, kwargs))
        return fut.result(timeout=timeout)


_worker = None
_worker_lock = threading.Lock()


def _get_worker():
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _Worker()
        return _worker


def run_on_page(cb, *args, timeout=180, **kwargs):
    """`cb(page, *args, **kwargs)` ifadesini ortak page ile isci thread'inde calistirir."""
    return _get_worker()._submit("on_page", cb, args, kwargs, timeout)


def run_with_browser(cb, *args, timeout=240, **kwargs):
    """`cb(browser, page, *args)` ifadesini ortak browser ile isci thread'inde calistirir."""
    return _get_worker()._submit("with_browser", cb, args, kwargs, timeout)