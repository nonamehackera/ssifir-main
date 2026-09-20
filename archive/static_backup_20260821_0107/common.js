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

// ============ MODEL DURUMU + SIDEBAR AKTIFLIK ============
function initChrome(activePage) {
  document.querySelectorAll('.nav-item').forEach(a => {
    a.classList.toggle('active', a.dataset.page === activePage);
  });
  API.get('/api/model/info').then(d => {
    const st = el('side-status');
    const md = el('side-model');
    if (d.status === 'ok') {
      st.textContent = '● Model hazır';
      st.classList.remove('err');
      md.textContent = 'Model: ' + (d.model_version || '—');
    } else if (d.status === 'loading') {
      el('side-status').textContent = '● Yükleniyor...';
      setTimeout(() => initChrome(activePage), 2500);
    } else {
      st.textContent = '● HATA';
      st.classList.add('err');
      md.textContent = d.error || 'unknown';
    }
  }).catch(() => {
    const st = el('side-status');
    st.textContent = '● Bağlantı yok';
    st.classList.add('err');
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
      div.className = 'search-item';
      div.innerHTML = `<div><div class="si-name">${t.name}</div><div class="si-league">${t.league || ''} · Son: ${t.form || '?'}</div></div><div class="si-elo">ELO ${Math.round(t.elo)}</div>`;
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

document.addEventListener('DOMContentLoaded', () => SearchModal.wire());
