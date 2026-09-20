// ============ ORTAK YARDIMCILAR ============
const API = {
  async json(url, opts) {
    const r = await fetch(url, opts);
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status));
    return d;
  },
  get(url) { return this.json(url); },
  post(url, body) {
    return this.json(url, { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  }
};

function toast(msg, isErr) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show' + (isErr ? ' err' : '');
  clearTimeout(t._t);
  t._t = setTimeout(() => t.className = 'toast', 2800);
}

function el(id) { return document.getElementById(id); }
function fmtPct(v) { return (Math.round(v * 10) / 10) + '%'; }
function setBar(id, pct, color) {
  const b = el(id);
  if (b) b.style.width = Math.max(0, Math.min(100, pct)) + '%';
  if (color && b) b.style.background = color;
}
function initials(name) {
  return (name || '?').trim().split(/\s+/).slice(0, 2).map(w => w[0]).join('').toUpperCase();
}

// ============ MODEL DURUMU + NAVBAR AKTIFLIK ============
function initChrome(activePage) {
  document.querySelectorAll('.nav-link').forEach(a => {
    a.classList.toggle('active', a.dataset.page === activePage);
  });
  API.get('/api/model/info').then(d => {
    const st = el('nav-status-txt');
    const md = el('nav-model');
    const dot = el('nav-dot');
    const dv = el('nav-data');
    if (d.status === 'ok') {
      st.textContent = '● Model hazır';
      md.textContent = 'Model: ' + (d.model_version || '—');
      // Veri guncluluk bilgisi
      if (d.freshness && dv) {
        const di = d.data_info || {};
        const fr = d.freshness;
        const dateStr = (di.newest_match_date || '?').slice(0, 10);
        dv.textContent = `Veri: ${dateStr} · ${di.n_current_teams || 0}/${di.n_teams || 0} güncel`;
        if (fr.needs_update) {
          dv.style.color = 'var(--loss)';
          dv.textContent += ' · ⚠ Guncelleme onerilir';
        } else {
          dv.style.color = 'var(--muted)';
          dv.textContent += ' · Guncel';
        }
        // Pazartesi otomatik kontrol uyarisi
        if (fr.is_monday && fr.needs_update && dot) {
          dot.className = 'dot'; // sari/uyari rengi (CSS'te .dot sorunsuz)
          dot.style.background = 'var(--draw)';
        }
      }
    } else if (d.status === 'loading') {
      el('nav-status-txt').textContent = '● Yükleniyor...';
      setTimeout(() => initChrome(activePage), 2500);
    } else {
      st.textContent = '● HATA';
      md.textContent = d.error || 'Model yüklenemedi';
    }
  }).catch(() => {
    const st = el('nav-status-txt');
    st.textContent = '● Bağlantı yok';
  });
}

// ============ TAKIM ARAMA MODALI ============
const SearchModal = (() => {
  let target = 'home';
  let onSelect = null;
  let cache = {};

  function open(t, cb) {
    target = t; onSelect = cb;
    el('search-bg').classList.add('active');
    el('search-input-modal').value = '';
    el('search-results-modal').innerHTML = '';
    el('search-input-modal').focus();
  }
  function close() { el('search-bg').classList.remove('active'); }

  function render(items) {
    const box = el('search-results-modal');
    box.innerHTML = '';
    if (!items.length) { box.innerHTML = '<div style="padding:20px;color:var(--muted)">Takım bulunamadı.</div>'; return; }
    items.forEach(t => {
      const div = document.createElement('div');
      const stale = t.current_data === false;
      const tag = stale ? ' <span style="color:var(--draw);font-size:10px;font-weight:700;opacity:.7">ESKI SEZON</span>' : '';
      div.className = 'search-item';
      div.innerHTML = `<div><div class="si-name">${t.name}${tag}</div><div class="si-league">${t.league || ''} · Son: ${t.form || '?'}</div></div><div class="si-elo">ELO ${Math.round(t.elo)}</div>`;
      div.onclick = () => { close(); onSelect && onSelect(t); };
      box.appendChild(div);
    });
  }

  async function query(q) {
    if (!q) { el('search-results-modal').innerHTML = ''; return; }
    if (cache[q]) return render(cache[q]);
    const data = await API.get('/api/search?q=' + encodeURIComponent(q));
    cache[q] = data;
    render(data);
  }

  function wire() {
    el('search-bg').addEventListener('click', e => { if (e.target === el('search-bg')) close(); });
    el('search-close').onclick = close;
    let to;
    el('search-input-modal').addEventListener('input', e => {
      clearTimeout(to);
      const q = e.target.value.trim();
      to = setTimeout(() => query(q), 160);
    });
  }
  return { open, close, wire };
})();

// ============ VERI GUNCELLEME ILERLEME CUBUGU ============
let _syncTimer = null;
function startSyncPolling() {
  if (_syncTimer) return; // zaten calisiyor
  console.log('[SYNC] polling basladi');
  _syncTimer = setInterval(async () => {
    try {
      const d = await API.get('/api/sync/progress');
      console.log('[SYNC] progress:', JSON.stringify(d).slice(0,120));
      const bar = document.getElementById('sync-bar');
      const fill = document.getElementById('sync-progress-fill');
      const pct = document.getElementById('sync-percent');
      const txt = document.getElementById('sync-text');
      if (!bar) return;
      if (d.running) {
        bar.style.display = 'flex';
        const p = Math.round(d.percent || 0);
        if (fill) fill.style.width = p + '%';
        if (pct) pct.textContent = p + '%';
        if (txt) txt.textContent = d.message || 'Veri güncelleniyor...';
      } else {
        if (d.stage === 'done' && d.updated_at) {
          const ago = (Date.now() - new Date(d.updated_at).getTime()) / 1000;
          if (ago < 4) {
            bar.style.display = 'flex';
            if (fill) fill.style.width = '100%';
            if (pct) pct.textContent = '100%';
            if (txt) txt.textContent = d.message || 'Tamamlandi';
            return;
          }
        }
        bar.style.display = 'none';
        if (_syncTimer) { clearInterval(_syncTimer); _syncTimer = null; }
      }
    } catch (e) { /* sessiz */ }
  }, 2000);
}

function triggerDataSync() {
  API.post('/api/sync/start').then(d => {
    if (d.started) {
      toast('Veri güncelleme başladı');
      startSyncPolling();
    } else {
      toast(d.message || 'Başlatılamadı', true);
    }
  }).catch(e => toast('Hata: ' + e.message, true));
}

// Sayfa yuklendiginde ilerleme poll'unu baslat
document.addEventListener('DOMContentLoaded', () => {
  SearchModal.wire();
  startSyncPolling();
});
