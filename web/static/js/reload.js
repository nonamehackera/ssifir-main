/* FTMS - Paylasilan "Yenile" mantigi
   Backend /api/reload ile gold veriden takim+lig indexini tazeler,
   sonra sayfayi HARD REFRESH yapar (cache'li eski CSS/JS'i atar).
   Boylece "yenileyemiyoruz / eski tasarim kaliyor" sorunu cozulur. */

let _reloading = false;

function ftmsReload(btn, reloadPage = true) {
  if (_reloading) return;
  _reloading = true;
  const original = btn ? btn.textContent : '';
  if (btn) { btn.disabled = true; btn.textContent = '⟳ Yenileniyor...'; }
  fetch('/api/reload', { method: 'POST', headers: { 'Content-Type': 'application/json' } })
    .then(r => r.json())
    .then(d => {
      if (d.status === 'ok') {
        if (btn) { btn.textContent = '✓ ' + d.teams + ' takim'; }
        if (reloadPage) {
          // Gercek hard refresh: tarayiciya tum cache'i atlatip
          // sayfayi + statik dosyalari bambasqa versiyonla yeniden getir.
          const v = Date.now();
          const url = new URL(location.href);
          url.searchParams.set('_rb', v);
          // location.reload(true) cesitli tarayicilarda hard reload yapar
          setTimeout(() => {
            try { location.reload(true); } catch (e) { location.href = url.toString(); }
          }, 350);
        } else {
          if (btn) setTimeout(() => { btn.textContent = original; btn.disabled = false; }, 1200);
          _reloading = false;
        }
      } else {
        alert('Yenileme hatasi: ' + (d.error || 'bilinmiyor'));
        if (btn) { btn.textContent = original; btn.disabled = false; }
        _reloading = false;
      }
    })
    .catch(e => {
      alert('Yenileme hatasi: ' + e);
      if (btn) { btn.textContent = original; btn.disabled = false; }
      _reloading = false;
    });
}

function addReloadButton(containerId, btnId, label) {
  const c = document.getElementById(containerId);
  if (!c) return;
  const btn = document.createElement('button');
  btn.id = btnId;
  btn.className = 'reload-btn';
  btn.textContent = label || '⟳ Yenile';
  btn.onclick = function () { ftmsReload(this, true); };
  c.appendChild(btn);
}

/* Tarayici acilisinda eger eski bir cache-bust param varsa temizle
   (URL'de ?_rb=... kalmasin diye) */
(function () {
  const url = new URL(location.href);
  if (url.searchParams.has('_rb')) {
    url.searchParams.delete('_rb');
    history.replaceState(null, '', url.toString());
  }
})();
