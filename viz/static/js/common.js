/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Shared helpers for both viewers: DOM building, formatting, fetch, and the
   playback clock that drives the timelines. */

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Build an element. Children may be nodes, strings, or nested arrays. */
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === 'dataset') {
      Object.assign(node.dataset, value);
    } else node.setAttribute(key, value);
  }
  for (const child of [children].flat(4)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

// ---------------------------------------------------------------- fetch

export async function api(path, options) {
  const response = await fetch(path, options);
  const text = await response.text();
  let payload;
  try { payload = JSON.parse(text); } catch { payload = { error: text.slice(0, 400) }; }
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

export async function post(path, body) {
  return api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
}

let toastTimer = null;
export function toast(message, isError = false) {
  const existing = $('.toast');
  if (existing) existing.remove();
  const node = el('div', { class: `toast${isError ? ' err' : ''}`, text: message });
  document.body.append(node);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.remove(), isError ? 7000 : 3200);
}

// ---------------------------------------------------------------- format

export function num(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const n = Number(value);
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e4) return (n / 1e3).toFixed(0) + 'k';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(n);
}

export function usd(value) {
  const n = Number(value || 0);
  return n >= 100 ? `$${n.toFixed(0)}` : n >= 1 ? `$${n.toFixed(2)}` : `$${n.toFixed(3)}`;
}

export function dur(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

export function durMs(ms) {
  if (ms === null || ms === undefined) return '—';
  return ms < 1000 ? `${Math.round(ms)}ms` : dur(ms / 1000);
}

export function bytes(value) {
  const n = Number(value || 0);
  if (n >= 1e6) return (n / 1e6).toFixed(1) + ' MB';
  if (n >= 1e3) return (n / 1e3).toFixed(0) + ' KB';
  return n + ' B';
}

export function clockOf(ms, base) {
  if (!ms || !base) return '00:00';
  const s = Math.max(0, Math.round((ms - base) / 1000));
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

export function shortDate(iso) {
  if (!iso) return '—';
  return String(iso).replace('T', ' ').replace(/\.\d+.*$/, '').slice(0, 16);
}

export function truthy(value) {
  return value === true ? '✓' : value === false ? '✗' : '—';
}

const LANE_COLORS = ['--a1', '--a2', '--a3', '--a4', '--a5', '--a6'];
export function laneColor(index) {
  const style = getComputedStyle(document.documentElement);
  return style.getPropertyValue(LANE_COLORS[index % LANE_COLORS.length]).trim() || '#5aa9ff';
}

/** Deterministic colour for an agent id, so the same agent looks the same everywhere. */
export function agentColor(id) {
  const hue = (Number(id) * 47) % 360;
  return `hsl(${hue} 62% 62%)`;
}

export function initials(name) {
  const clean = String(name || '?').replace(/[^A-Za-z0-9]+/g, ' ').trim();
  const words = clean.split(/\s+/);
  return ((words[0] || '?')[0] + (words[1] ? words[1][0] : '')).toUpperCase();
}

// ---------------------------------------------------------------- rendering bits

/** Render a unified diff patch with per-line colouring. */
export function renderDiff(patch) {
  const box = el('div', { class: 'diff' });
  for (const line of String(patch || '').split('\n')) {
    let cls = 'ctx';
    if (line.startsWith('+++') || line.startsWith('---')) cls = 'hunk';
    else if (line.startsWith('@@')) cls = 'hunk';
    else if (line.startsWith('+')) cls = 'add';
    else if (line.startsWith('-')) cls = 'del';
    box.append(el('span', { class: cls, text: line || ' ' }));
  }
  return box;
}

export function fold(title, body, open = false) {
  const node = el('details', { class: 'fold' }, [
    el('summary', {}, [title]),
    el('div', { class: 'fold-body' }, [body]),
  ]);
  if (open) node.open = true;
  return node;
}

export function kv(pairs) {
  const list = el('dl', { class: 'kv' });
  for (const [key, value] of pairs) {
    if (value === null || value === undefined || value === '') continue;
    list.append(el('dt', { text: key }));
    list.append(el('dd', {}, [value.nodeType ? value : String(value)]));
  }
  return list;
}

export function chip(text, cls = '') {
  return el('span', { class: `chip ${cls}`.trim() }, [text]);
}

/** A headline number with its label.
 *
 *  `title` matters more than it looks: a bare number in a header is exactly the
 *  thing a reader will guess the provenance of, and guess wrong. Anything whose
 *  source is not obvious from the label should say so on hover. */
export function stat(value, label, { title = '', onclick = null } = {}) {
  const attrs = { class: `stat${onclick ? ' clickable' : ''}` };
  if (title) attrs.title = title;
  if (onclick) attrs.onclick = onclick;
  return el('div', attrs, [
    el('div', { class: 'v', text: value }),
    el('div', { class: 'k', text: label }),
  ]);
}

export function empty(message) {
  return el('div', { class: 'empty', text: message });
}

// ---------------------------------------------------------------- playback

/**
 * A clock that walks a numeric range and calls back on every frame.
 *
 * Both viewers replay a recording, but on different units -- the founder timeline
 * runs on epoch milliseconds, the crowd on an integer round counter -- so the
 * clock is unitless and the caller decides what a "second" of playback means.
 */
export class Playback {
  constructor({ onTick, onState }) {
    this.start = 0;
    this.end = 1;
    this.value = 0;
    this.speed = 1;
    this.playing = false;
    this.unitsPerSecond = 1;
    this.onTick = onTick || (() => {});
    this.onState = onState || (() => {});
    this._raf = null;
    this._last = 0;
  }

  setRange(start, end, unitsPerSecond) {
    this.start = start;
    this.end = Math.max(end, start + 1);
    this.unitsPerSecond = unitsPerSecond || 1;
    this.value = start;
    this.emit();
  }

  seek(value) {
    this.value = Math.min(this.end, Math.max(this.start, value));
    this.emit();
  }

  /** Fraction 0..1 of the way through the range. */
  get progress() { return (this.value - this.start) / (this.end - this.start); }

  seekFraction(fraction) { this.seek(this.start + fraction * (this.end - this.start)); }

  play() {
    if (this.playing) return;
    // Restarting from the end is what a viewer means by "play" after a run
    // finished, rather than sitting still.
    if (this.value >= this.end) this.value = this.start;
    this.playing = true;
    this._last = performance.now();
    this.onState(this);
    const step = (now) => {
      if (!this.playing) return;
      const delta = (now - this._last) / 1000;
      this._last = now;
      this.value += delta * this.unitsPerSecond * this.speed;
      if (this.value >= this.end) {
        this.value = this.end;
        this.pause();
      }
      this.emit();
      if (this.playing) this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }

  pause() {
    this.playing = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
    this.onState(this);
  }

  toggle() { this.playing ? this.pause() : this.play(); }

  setSpeed(speed) { this.speed = speed; this.onState(this); }

  emit() { this.onTick(this.value, this); this.onState(this); }
}

/** Read ?key=value from the address bar, and write it back without navigating. */
export const params = {
  get(key) { return new URLSearchParams(location.search).get(key); },
  set(key, value) {
    const next = new URLSearchParams(location.search);
    if (value) next.set(key, value); else next.delete(key);
    history.replaceState(null, '', `${location.pathname}?${next.toString()}`);
  },
};
