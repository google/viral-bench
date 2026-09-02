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

/* Founder trajectory viewer.
 *
 * A build is a sequence of turns, each turn a transcript, each transcript a stream
 * of tool calls and assistant text on an epoch-millisecond clock. The turns of one
 * build never overlap -- the agents run serially -- so a single sorted timeline is
 * a faithful replay of the run, and everything on screen is a projection of one
 * number: the playback clock.
 *
 * The three run structures differ in what the lanes mean, so the header and the
 * mode panel change with the structure while the feed and the timeline do not.
 */

import {
  $, api, agentColor, bytes, chip, clear, clockOf, dur, durMs, el, empty, fold,
  kv, laneColor, num, params, Playback, post, renderDiff, shortDate, stat, toast, truthy, usd,
} from './common.js';

const state = {
  data: null,
  events: [],
  selected: null,
  tab: 'overview',
  laneFilter: null,
  groupFilter: null,
  // Step rows carry the per-call cost and reasoning-token counts, but there is one
  // for every model call, so on by default they are half the feed and drown the
  // actual work. Off until asked for.
  showSteps: false,
  revealed: 0,
  app: null,
};

const ICONS = {
  browser: '🌐', edit: '✎', shell: '$', read: '📄', search: '🔍',
  task: '⑂', skill: '★', todo: '☑', web: '⚙', other: '•',
};

/** Non-tool event kinds, in the order the filter bar offers them. */
const KIND_META = {
  reasoning: { icon: '🧠', label: 'thinking', color: 'var(--purple)' },
  prompt: { icon: '📥', label: 'prompts', color: 'var(--cyan)' },
  text: { icon: '💬', label: 'narration', color: 'var(--fg)' },
  patch: { icon: '⌥', label: 'patches', color: 'var(--ok)' },
};

/** First line of an error worth putting in a one-line summary.
 *  Playwright failures arrive as markdown whose first line is a "### Result"
 *  heading, so taking line 0 shows every failure as the same useless string. */
function firstMeaningfulLine(text) {
  for (const line of String(text || '').split('\n')) {
    const trimmed = line.trim();
    if (trimmed && !trimmed.startsWith('#') && !trimmed.startsWith('```')) return trimmed;
  }
  return String(text || '').split('\n')[0] || '';
}

/** Drop the build-root prefix so a file argument reads as `src/builder.py`.
 *  Every path an agent touches is absolute and ~110 characters of it is the same
 *  build directory, which pushes the part that identifies the file off-screen. */
/** Where to fetch one event's full record.
 *  A traced build addresses events by part_id in the session dumps, while a build old
 *  enough to have no dump still addresses them by transcript file and line. */
function eventUrl(e) {
  const base = `/api/founder/${encodeURIComponent(state.data.build_id)}/event`;
  return e.part_id
    ? `${base}?part=${encodeURIComponent(e.part_id)}`
    : `${base}?phase=${encodeURIComponent(e.phase)}&line=${e.line}`;
}

function shortPath(text) {
  const root = state.data?.paths?.root;
  if (!text || !root) return text;
  return String(text).split(`${root}/app/`).join('').split(`${root}/`).join('');
}

const playback = new Playback({ onTick: onClock, onState: renderTransport });

// ---------------------------------------------------------------- loading

async function load(buildId) {
  if (!buildId) return;
  $('#feed').replaceChildren(el('div', { class: 'empty' }, [el('span', { class: 'spin', text: '◐' }), ' loading…']));
  try {
    const data = await api(`/api/founder/${encodeURIComponent(buildId)}`);
    state.data = data;
    state.events = data.events || [];
    state.selected = null;
    state.laneFilter = null;
    state.groupFilter = null;
    state.tab = 'overview';
    params.set('build', buildId);
    $('#buildInput').value = buildId;
    for (const id of ['dlJson', 'dlZip']) $(`#${id}`).disabled = false;
    $('#crowdBtn').disabled = !(data.crowd_runs || []).length;
    $('#crowdBtn').textContent = `Crowd runs (${(data.crowd_runs || []).length})`;

    renderHead();
    renderLanes();
    renderFilters();
    renderTabs();

    const span = data.span || {};
    if (span.start_ms && span.end_ms && span.end_ms > span.start_ms) {
      // One second of playback covers a minute of the build by default, because a
      // build runs ~45 minutes and nobody watches that in real time.
      playback.setRange(span.start_ms, span.end_ms, 60000);
      playback.seek(span.end_ms);   // land on the finished run; press play to replay
    } else {
      playback.setRange(0, 1, 1);
      playback.seek(1);
    }
    onClock(playback.value);
  } catch (err) {
    $('#feed').replaceChildren(el('div', { class: 'empty err-text', text: String(err.message || err) }));
    toast(String(err.message || err), true);
  }
}

// ---------------------------------------------------------------- header

function renderHead() {
  const d = state.data;
  const r = d.record || {};
  const t = d.totals || {};
  const cap = d.trace?.summary || {};
  const modeClass = { solo: 'accent', team: 'ok', dynamic: 'warn' }[d.mode] || '';
  const statusClass = r.status === 'ok' ? 'ok' : r.status ? 'err' : '';

  const chips = [
    chip(d.mode_label, `mode ${modeClass}`),
    chip(r.status || 'unknown', statusClass),
    r.model && chip(r.model.split('/').pop()),
    d.manifest?.app_type && chip(d.manifest.app_type),
    r.collab && chip(`collab ${r.collab}`),
    r.rounds_run != null && chip(`${r.rounds_run} round${r.rounds_run === 1 ? '' : 's'}`),
    r.shipped_early && chip('shipped early', 'ok'),
    r.qa_verified === true && chip('QA verified', 'ok'),
    r.qa_verified === false && chip('QA not verified', 'warn'),
    t.tool_errors > 0 && chip(`${t.tool_errors} tool error${t.tool_errors === 1 ? '' : 's'}`, 'warn'),
    r.brief_fingerprint && chip(`brief ${String(r.brief_fingerprint).slice(0, 8)}`),
    cap.has_thinking && el('span', {
      class: 'chip purple', style: 'cursor:pointer',
      title: 'the model\'s own chain of thought. Click for the Thinking tab',
      onclick: () => { state.tab = 'thinking'; renderTabs(); },
    }, [`🧠 ${num(cap.stream_chars)} chars of thinking`
        + (cap.team_chars ? ` · ${num(cap.team_chars)} by the team` : '')
        + (cap.redacted ? ` · ${cap.redacted} redacted` : '')]),
    cap.prompts > 0 && el('span', {
      class: 'chip', style: 'cursor:pointer;color:#9be5e2',
      onclick: () => { state.tab = 'prompts'; renderTabs(); },
    }, [`📥 ${cap.prompts} prompt${cap.prompts === 1 ? '' : 's'}`]),
    t.subagent_sessions > 0 && chip(`⑂ ${t.subagent_sessions} subagent session${t.subagent_sessions === 1 ? '' : 's'}`, 'accent'),
    cap.source === 'transcript' && el('span', {
      class: 'chip warn', title: 'recorded before the pipeline captured thinking, prompts, subagents or patches',
    }, ['partial trace']),
  ].filter(Boolean);

  clear($('#head')).append(
    el('div', { style: 'display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap' }, [
      el('div', { style: 'flex:1 1 380px;min-width:0' }, [
        el('div', { style: 'display:flex;gap:9px;align-items:baseline;flex-wrap:wrap' }, [
          el('div', { style: 'font-size:16px;font-weight:700' }, [d.manifest?.title || r.idea_id || d.build_id]),
          el('div', { class: 'mono faint small', text: d.build_id }),
        ]),
        el('div', { class: 'chips', style: 'margin-top:6px' }, chips),
      ]),
      el('div', { class: 'stats' }, [
        stat(String(d.phases.length), 'turns'),
        stat(num(t.tools), 'tool calls'),
        stat(num(t.events), 'events'),
        stat(usd(t.cost), 'cost'),
        stat(num(t.tokens_total), 'tokens'),
        stat(dur(t.duration_s), 'wall clock'),
      ]),
    ]),
  );
}

// ---------------------------------------------------------------- swimlanes

function renderLanes() {
  const d = state.data;
  const span = d.span || {};
  const box = clear($('#lanes'));
  box.append(el('h2', {}, [
    'Swimlanes',
    el('span', { class: 'count' }, [
      ` · ${d.lanes.length} actor${d.lanes.length === 1 ? '' : 's'}, ${d.phases.length} turn${d.phases.length === 1 ? '' : 's'}`,
      d.mode === 'team'
        ? ' · one session id per lane means each specialist resumes its own memory each round'
        : '',
    ]),
  ]));

  const t0 = span.start_ms, t1 = span.end_ms;
  const width = (ms) => (!t0 || !t1 || t1 <= t0 ? 0 : ((ms - t0) / (t1 - t0)) * 100);
  const lanes = el('div', { class: 'lanes' });

  d.lanes.forEach((lane, index) => {
    const color = laneColor(index);
    const track = el('div', { class: 'lane-track', dataset: { lane: lane.key } });

    for (const phase of d.phases.filter((p) => lane.phases.includes(p.phase))) {
      // Filter by lane as well as phase. A delegated session is attributed to the
      // turn it ran inside, so keying on the phase alone would draw every
      // subagent as active for the whole turn -- which is the opposite of what
      // the spawn gantt below shows, and the opposite of the truth.
      const events = state.events.filter(
        (e) => e.phase === phase.phase && e.lane === lane.key && e.t);
      if (!events.length) continue;
      const start = events[0].t, end = events[events.length - 1].t;
      const left = width(start), right = width(end);
      const seg = el('div', {
        class: `seg${phase.ok === false ? ' failed' : ''}`,
        style: `left:${left}%;width:${Math.max(right - left, 0.6)}%;background:${color}`,
        title: `${phase.phase} · ${dur(phase.duration_s)} · ${events.length} events · rc=${phase.returncode}`,
        dataset: { phase: phase.phase, start: String(start) },
        onclick: () => { playback.pause(); playback.seek(start + 1); },
      });
      const label = lane.kind === 'subagent'
        ? durMs(end - start)
        : (d.mode === 'dynamic' ? `t${phase.round}`
          : (d.mode === 'solo' ? phase.phase : `r${phase.round}`));
      seg.append(el('div', { class: 'seg-label', text: label }));
      track.append(seg);
    }

    // A dot per tool call: the density of the run at a glance, and it makes an
    // agent that did nothing for two minutes obvious.
    for (const e of state.events) {
      if (e.kind !== 'tool' || e.lane !== lane.key || !e.t) continue;
      track.append(el('div', {
        class: 'evdot',
        style: `left:${width(e.t)}%;background:${e.status === 'error' ? 'var(--err)' : '#fff'};opacity:${e.status === 'error' ? 1 : .5}`,
      }));
    }

    track.append(el('div', { class: 'tick', id: `tick-${lane.key}`, style: 'left:0%' }));
    track.addEventListener('click', (event) => {
      if (event.target !== track) return;
      const rect = track.getBoundingClientRect();
      playback.pause();
      playback.seekFraction((event.clientX - rect.left) / rect.width);
    });

    lanes.append(el('div', { class: 'lane' }, [
      el('div', { class: 'lane-name', title: lane.session_ids.join('\n') }, [
        el('span', { class: 'lane-dot', style: `background:${color}` }),
        lane.label,
        el('span', { class: 'faint small mono' }, [` ${lane.session_id ? lane.session_id.slice(0, 10) : ''}`]),
      ]),
      track,
    ]));
  });

  box.append(lanes);

  if (d.mode === 'dynamic' && (d.spawns || []).length) box.append(renderSpawnGantt());
}

/** Dynamic mode: the invented subagents, drawn on their real overlapping intervals. */
function renderSpawnGantt() {
  const spawns = state.data.spawns.filter((s) => s.start_ms && s.end_ms);
  if (!spawns.length) return el('div');
  const t0 = Math.min(...spawns.map((s) => s.start_ms));
  const t1 = Math.max(...spawns.map((s) => s.end_ms));
  const pct = (ms) => ((ms - t0) / Math.max(t1 - t0, 1)) * 100;
  const peak = state.data.orchestration?.peak_concurrent_subagents;

  const rows = spawns.map((spawn, index) => {
    const color = agentColor(index * 3 + 5);
    return el('div', { class: 'lane' }, [
      el('div', { class: 'lane-name', title: spawn.prompt }, [
        el('span', { class: 'lane-dot', style: `background:${color}` }),
        el('span', { class: 'small', text: spawn.label.slice(0, 34) }),
      ]),
      el('div', { class: 'lane-track', style: 'height:18px' }, [
        el('div', {
          class: `seg done${spawn.status === 'error' ? ' failed' : ''}`,
          style: `left:${pct(spawn.start_ms)}%;width:${Math.max(pct(spawn.end_ms) - pct(spawn.start_ms), 0.8)}%;background:${color}`,
          title: `${spawn.subagent_type} · ${durMs(spawn.end_ms - spawn.start_ms)} · ${spawn.status}${spawn.resumed ? ' · resumed' : ''}`,
        }, [el('div', { class: 'seg-label', text: durMs(spawn.end_ms - spawn.start_ms) })]),
      ]),
    ]);
  });

  return el('div', { style: 'margin-top:12px' }, [
    el('h2', {}, ['Subagents the orchestrator invented',
      el('span', { class: 'count' }, [` · ${spawns.length} spawn${spawns.length === 1 ? '' : 's'}`,
        peak ? `, up to ${peak} running at once` : '']),
    ]),
    el('div', { class: 'lanes' }, rows),
  ]);
}

// ---------------------------------------------------------------- transport

function renderTransport() {
  const box = $('#transport');
  const span = state.data?.span || {};
  if (!box.dataset.built) {
    box.dataset.built = '1';
    clear(box).append(el('div', { class: 'transport' }, [
      el('button', { id: 'playBtn', class: 'primary', onclick: () => playback.toggle() }, ['▶ Play']),
      el('button', { onclick: () => { playback.pause(); playback.seek(playback.start); } }, ['⏮']),
      el('button', { onclick: () => step(-1) }, ['◀ step']),
      el('button', { onclick: () => step(1) }, ['step ▶']),
      el('select', {
        id: 'speedSel',
        onchange: (e) => playback.setSpeed(Number(e.target.value)),
      }, [1, 2, 4, 8, 16].map((s) => el('option', { value: String(s), text: `${s}×` }))),
      el('input', {
        type: 'range', class: 'scrub', id: 'scrub', min: '0', max: '1000', value: '1000',
        oninput: (e) => { playback.pause(); playback.seekFraction(Number(e.target.value) / 1000); },
      }),
      el('span', { class: 'clock', id: 'clockLabel' }),
    ]));
  }
  const play = $('#playBtn');
  if (play) play.textContent = playback.playing ? '❚❚ Pause' : '▶ Play';
  const scrub = $('#scrub');
  if (scrub && document.activeElement !== scrub) scrub.value = String(Math.round(playback.progress * 1000));
  const label = $('#clockLabel');
  if (label) {
    label.textContent = `${clockOf(playback.value, span.start_ms)} / ${clockOf(span.end_ms, span.start_ms)}`
      + `  ·  ${state.revealed}/${state.events.length} events`;
  }
}

function step(direction) {
  playback.pause();
  const now = playback.value;
  const times = state.events.map((e) => e.t).filter(Boolean);
  const next = direction > 0
    ? times.find((t) => t > now + 0.5)
    : [...times].reverse().find((t) => t < now - 0.5);
  if (next) playback.seek(next);
}

/** The single place the clock turns into what is on screen. */
function onClock(value) {
  if (!state.data) return;
  const revealed = state.events.filter((e) => (e.t || 0) <= value);
  const changed = revealed.length !== state.revealed;
  state.revealed = revealed.length;

  const span = state.data.span || {};
  const fraction = (!span.start_ms || !span.end_ms || span.end_ms <= span.start_ms)
    ? 1 : (value - span.start_ms) / (span.end_ms - span.start_ms);
  for (const lane of state.data.lanes) {
    const tick = document.getElementById(`tick-${lane.key}`);
    if (tick) tick.style.left = `${Math.min(100, Math.max(0, fraction * 100))}%`;
  }
  for (const seg of document.querySelectorAll('.seg[data-start]')) {
    seg.classList.toggle('done', Number(seg.dataset.start) <= value);
  }
  if (changed) renderFeed(revealed);
  renderTransport();
}

// ---------------------------------------------------------------- feed

function renderFilters() {
  const d = state.data;
  const box = clear($('#feedFilters'));
  const mk = (label, active, onclick) =>
    el('span', { class: `chip${active ? ' accent' : ''}`, style: 'cursor:pointer', onclick }, [label]);

  box.append(mk('all lanes', !state.laneFilter, () => { state.laneFilter = null; refresh(); }));
  d.lanes.forEach((lane, index) => box.append(el('span', {
    class: `chip${state.laneFilter === lane.key ? ' accent' : ''}`,
    style: 'cursor:pointer',
    onclick: () => { state.laneFilter = state.laneFilter === lane.key ? null : lane.key; refresh(); },
  }, [el('span', { class: 'lane-dot', style: `background:${laneColor(index)}` }), lane.label])));

  box.append(el('span', { class: 'faint', style: 'margin:0 4px' }, ['|']));
  box.append(mk('all kinds', !state.groupFilter, () => { state.groupFilter = null; refresh(); }));
  // The new event kinds first: thinking and prompts are the reason this build
  // records more than it used to, and burying them under the tool families
  // would make them look like an afterthought.
  for (const [kind, meta] of Object.entries(KIND_META)) {
    const count = d.totals[kind === 'text' ? 'texts' : kind === 'patch' ? 'patches' : kind] || 0;
    if (!count) continue;
    box.append(el('span', {
      class: `chip${state.groupFilter === `k:${kind}` ? ' accent' : ''}`,
      style: `cursor:pointer;color:${meta.color}`,
      onclick: () => { state.groupFilter = state.groupFilter === `k:${kind}` ? null : `k:${kind}`; refresh(); },
    }, [`${meta.icon} ${meta.label} ${count}`]));
  }
  for (const [group, count] of Object.entries(d.group_counts).sort((a, b) => b[1] - a[1])) {
    box.append(mk(`${ICONS[group] || '•'} ${group} ${count}`, state.groupFilter === group,
      () => { state.groupFilter = state.groupFilter === group ? null : group; refresh(); }));
  }
  box.append(mk('errors', state.groupFilter === '__err',
    () => { state.groupFilter = state.groupFilter === '__err' ? null : '__err'; refresh(); }));
  box.append(el('span', { class: 'faint', style: 'margin:0 4px' }, ['|']));
  box.append(mk(`${state.showSteps ? '☑' : '☐'} model steps`, state.showSteps,
    () => { state.showSteps = !state.showSteps; refresh(); }));
}

function refresh() {
  renderFilters();
  renderFeed(state.events.filter((e) => (e.t || 0) <= playback.value));
}

function visible(events) {
  return events.filter((e) => {
    if (e.kind === 'step' && !state.showSteps) return false;
    if (state.laneFilter && e.lane !== state.laneFilter) return false;
    if (state.groupFilter === '__err') return e.status === 'error' || e.kind === 'error';
    if (state.groupFilter?.startsWith('k:')) return e.kind === state.groupFilter.slice(2);
    if (state.groupFilter && e.group !== state.groupFilter) return false;
    return true;
  });
}

function renderFeed(revealed) {
  const rows = visible(revealed);
  const box = $('#feed');
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
  $('#feedCount').textContent = `· ${rows.length} shown of ${state.events.length}`;

  clear(box);
  if (!rows.length) { box.append(empty('Nothing yet at this point in the run.')); return; }

  const base = state.data.span?.start_ms;
  // The tail is what a replay is watching, and a long build's head is reachable by
  // scrubbing rather than by scrolling through thousands of rows.
  const slice = rows.slice(-1200);
  if (rows.length > slice.length) {
    box.append(el('div', { class: 'empty', text: `…${rows.length - slice.length} earlier events (scrub back to see them)` }));
  }
  const frag = document.createDocumentFragment();
  for (const e of slice) frag.append(eventRow(e, base));
  box.append(frag);
  if (playback.playing || atBottom) box.scrollTop = box.scrollHeight;
}

function eventRow(e, base) {
  const laneIndex = state.data.lanes.findIndex((l) => l.key === e.lane);
  const color = laneColor(Math.max(laneIndex, 0));
  const row = el('div', {
    class: `ev kind-${e.kind}${e.status === 'error' ? ' status-error' : ''}${state.selected === e.i ? ' sel' : ''}`,
    style: `border-left-color:${state.selected === e.i ? 'var(--accent)' : color}`,
    dataset: { i: String(e.i) },
    onclick: () => select(e.i),
  });
  row.append(el('div', { class: 't', text: clockOf(e.t, base) }));

  if (e.kind === 'tool') {
    row.append(el('div', { class: 'ic', text: ICONS[e.group] || '•' }));
    row.append(el('div', { class: 'body' }, [
      el('div', { class: 'line1' }, [
        el('span', { class: 'tool', style: `color:${e.status === 'error' ? 'var(--err)' : color}`, text: e.tool.replace(/^browser_browser_/, 'browser.') }),
        el('span', { class: 'arg', title: e.summary || '', text: shortPath(e.summary || '') }),
        e.duration_ms != null && el('span', { class: 'badge', text: durMs(e.duration_ms) }),
        e.diff && el('span', { class: 'badge g-edit', text: `+${e.diff.additions ?? 0}/−${e.diff.deletions ?? 0}` }),
        e.exit != null && e.exit !== 0 && el('span', { class: 'badge', style: 'color:var(--err)', text: `exit ${e.exit}` }),
      ].filter(Boolean)),
      e.status === 'error' && e.error && el('div', { class: 'arg err-text', text: firstMeaningfulLine(e.error) }),
    ].filter(Boolean)));
  } else if (e.kind === 'reasoning') {
    row.append(el('div', { class: 'ic', text: '🧠' }));
    row.append(el('div', { class: 'body' }, [
      el('div', { class: 'line1' }, [
        el('span', { class: 'tool', style: 'color:var(--purple)', text: 'thinking' }),
        e.redacted
          ? el('span', { class: 'badge', style: 'color:var(--warn)', title: 'the model thought and the provider encrypted it', text: 'redacted' })
          : el('span', { class: 'badge', text: `${num(e.chars)} chars` }),
      ]),
      e.redacted
        ? el('div', { class: 'text-preview faint', text: 'The provider returned this thought encrypted, so it cannot be read. It is not an absence of thinking.' })
        : el('div', { class: 'text-preview', style: 'color:#d6bcff', text: e.text.slice(0, 400) }),
    ]));
  } else if (e.kind === 'prompt') {
    row.append(el('div', { class: 'ic', text: '📥' }));
    row.append(el('div', { class: 'body' }, [
      el('div', { class: 'line1' }, [
        el('span', { class: 'tool', style: 'color:var(--cyan)', text: 'prompt' }),
        el('span', { class: 'badge', text: `${num(e.chars)} chars` }),
        el('span', { class: 'badge', text: 'prompt text' }),
      ]),
      el('div', { class: 'text-preview', style: 'color:#9be5e2', text: e.text.slice(0, 400) }),
    ]));
  } else if (e.kind === 'patch') {
    row.append(el('div', { class: 'ic', text: '⌥' }));
    row.append(el('div', { class: 'body' }, [
      el('div', { class: 'line1' }, [
        el('span', { class: 'tool', style: 'color:var(--ok)', text: 'patch' }),
        el('span', { class: 'arg', text: (e.files || []).map(shortPath).join(', ') }),
        el('span', { class: 'badge g-edit', text: `${e.n_files} file${e.n_files === 1 ? '' : 's'}` }),
      ]),
    ]));
  } else if (e.kind === 'text') {
    row.append(el('div', { class: 'ic', text: '💬' }));
    row.append(el('div', { class: 'body' }, [
      el('div', { class: 'line1' }, [el('span', { class: 'tool', style: `color:${color}`, text: 'assistant' }),
        el('span', { class: 'badge', text: `${num(e.chars)} chars` })]),
      el('div', { class: 'text-preview', text: e.text.slice(0, 400) }),
    ]));
  } else if (e.kind === 'step') {
    row.append(el('div', { class: 'ic', text: '·' }));
    row.append(el('div', { class: 'body' }, [
      el('div', { class: 'line1' }, [
        el('span', { class: 'arg faint', text: `step · ${e.reason || ''} · ${usd(e.cost)} · ${num(e.tokens.total)} tok` }),
        e.tokens.reasoning > 0 && el('span', { class: 'badge', title: 'reasoning tokens billed for this step', text: `🧠 ${num(e.tokens.reasoning)}` }),
      ].filter(Boolean)),
    ]));
  } else {
    row.append(el('div', { class: 'ic', text: '⚠' }));
    row.append(el('div', { class: 'body' }, [
      el('div', { class: 'line1' }, [el('span', { class: 'tool err-text', text: e.name || 'error' }),
        el('span', { class: 'arg', text: e.message || '' })]),
    ]));
  }
  return row;
}

function select(index) {
  state.selected = index;
  state.tab = 'event';
  renderTabs();
  renderFeed(state.events.filter((e) => (e.t || 0) <= playback.value));
  const node = document.querySelector(`.ev[data-i="${index}"]`);
  if (node) node.scrollIntoView({ block: 'nearest' });
}

// ---------------------------------------------------------------- side tabs

function renderTabs() {
  const d = state.data;
  const t = d.totals || {};
  const tabs = [
    ['event', 'Event'],
    ['overview', 'Overview'],
    ['thinking', t.reasoning ? `Thinking (${t.reasoning})` : 'Thinking'],
    ['prompts', t.prompts ? `Prompts (${t.prompts})` : 'Prompts'],
    ['turns', 'Turns'],
    ['files', 'Files'],
    ['tools', 'Tools'],
  ];
  if ((d.trace?.sessions || []).length) {
    tabs.push(['sessions', `Sessions (${d.trace.sessions.length})`]);
  }
  if (d.mode === 'dynamic') tabs.push(['dynamic', 'Orchestration']);
  if ((d.screenshots || []).length) tabs.push(['shots', `Screenshots (${d.screenshots.length})`]);
  tabs.push(['app', 'Run the app']);
  tabs.push(['paths', 'Files on disk']);

  clear($('#tabs')).append(...tabs.map(([key, label]) =>
    el('div', { class: `tab${state.tab === key ? ' active' : ''}`, onclick: () => { state.tab = key; renderTabs(); } }, [label])));
  renderTabBody();
}

function renderTabBody() {
  const body = clear($('#tabBody'));
  const d = state.data;
  if (!d) return;
  const views = {
    event: viewEvent, overview: viewOverview, thinking: viewThinking, turns: viewTurns,
    files: viewFiles, tools: viewTools, dynamic: viewDynamic,
    shots: viewShots, app: viewApp, paths: viewPaths,
    prompts: viewPrompts, sessions: viewSessions,
  };
  body.append((views[state.tab] || viewOverview)());
}

function viewEvent() {
  if (state.selected === null) return empty('Click an event in the feed to inspect it.');
  const e = state.events[state.selected];
  if (!e) return empty('That event is gone.');
  const box = el('div', { class: 'detail' });

  box.append(el('div', { class: 'detail-head' }, [
    el('div', { style: 'display:flex;gap:8px;align-items:baseline;flex-wrap:wrap' }, [
      el('b', { text: e.kind === 'tool' ? e.tool : e.kind }),
      chip(e.phase),
      e.status && chip(e.status, e.status === 'error' ? 'err' : 'ok'),
      e.duration_ms != null && chip(durMs(e.duration_ms)),
      e.thought_signature && chip(`🧠 thought signature (${e.thought_signature.join(', ')})`, 'purple'),
    ].filter(Boolean)),
  ]));

  if (e.kind === 'text') {
    box.append(el('div', { style: 'padding:10px 12px' }, [el('pre', { class: 'block tall', text: e.text })]));
    if (e.truncated) box.append(loadFull(e));
    return box;
  }
  if (e.kind === 'step') {
    box.append(kv([
      ['reason', e.reason], ['cost', usd(e.cost)],
      ['tokens in', num(e.tokens.input)], ['tokens out', num(e.tokens.output)],
      ['reasoning tokens', num(e.tokens.reasoning)], ['total', num(e.tokens.total)],
      ['cache read', num(e.tokens.cache_read)], ['cache write', num(e.tokens.cache_write)],
      ['app snapshot', e.snapshot],
    ]));
    return box;
  }
  if (e.kind === 'error') {
    box.append(kv([['name', e.name], ['message', e.message]]));
    return box;
  }

  box.append(kv([
    ['tool', e.tool], ['group', e.group], ['call id', e.call_id],
    ['session', e.session], ['title', e.title],
    ['started', e.start_ms ? new Date(e.start_ms).toISOString().slice(11, 23) : null],
  ]));

  if (e.error) box.append(el('div', { style: 'padding:0 12px 8px' }, [el('pre', { class: 'block err-text', text: e.error })]));
  if (e.input_preview) {
    box.append(fold(`Input${e.input_truncated ? ` (first ${e.input_preview.length} of ${num(e.input_chars)} chars)` : ''}`,
      el('pre', { class: 'block', text: e.input_preview }), true));
  }
  if (e.diff) {
    box.append(fold(`Diff · ${e.diff.file || e.file || ''}  +${e.diff.additions ?? 0} / −${e.diff.deletions ?? 0}`,
      el('div', { class: 'faint small', text: 'Loading patch…' }), true));
    loadPatch(e, box.lastChild.querySelector('.fold-body'));
  }
  if (e.browser) {
    const b = e.browser;
    if (b.code) box.append(fold('Playwright code that ran', el('pre', { class: 'block', text: b.code }), true));
    if (b.url || b.title) box.append(kv([['page title', b.title], ['page url', b.url]]));
    if (b.result) box.append(fold('Result', el('pre', { class: 'block', text: b.result }), true));
    if (b.new_console_messages) {
      box.append(fold('Console messages ⚠', el('pre', { class: 'block err-text', text: b.new_console_messages }), true));
    }
    if (b.snapshot) {
      box.append(fold(`Accessibility snapshot${b.snapshot_truncated ? ` (first 4k of ${num(b.snapshot_chars)} chars)` : ''}`,
        el('pre', { class: 'block tall', text: b.snapshot })));
    }
  }
  if (e.screenshot_path) {
    const name = e.screenshot_path.split('/').pop();
    box.append(fold(`Screenshot · ${name}`, el('img', {
      class: 'shot', src: `/api/founder/${encodeURIComponent(state.data.build_id)}/shot/${encodeURIComponent(name)}`,
      onerror: (ev) => { ev.target.replaceWith(el('div', { class: 'faint small', text: `not on disk: ${e.screenshot_path}` })); },
    }), true));
  }
  if (e.todos?.length) {
    box.append(fold(`Task list (${e.todos.length})`, el('div', { style: 'font-size:11px' },
      e.todos.map((t) => el('div', { text: `${t.status === 'completed' ? '☑' : t.status === 'in_progress' ? '◐' : '☐'} ${t.content}` }))), true));
  }
  if (e.output_preview) {
    box.append(fold(`Output${e.output_truncated ? ` (first ${e.output_preview.length} of ${num(e.output_chars)} chars)` : ''}`,
      el('pre', { class: 'block tall', text: e.output_preview }), !e.browser));
  }
  if (e.output_truncated || e.input_truncated) box.append(loadFull(e));
  return box;
}

function loadFull(e) {
  return el('div', { style: 'padding:0 12px 12px' }, [
    el('button', {
      onclick: async (ev) => {
        ev.target.disabled = true;
        ev.target.textContent = 'loading…';
        try {
          const full = await api(eventUrl(e));
          ev.target.replaceWith(el('pre', { class: 'block tall', text: JSON.stringify(full.part ?? full, null, 2) }));
        } catch (err) { toast(String(err.message || err), true); ev.target.disabled = false; ev.target.textContent = 'retry'; }
      },
    }, ['Load the full record from disk']),
  ]);
}

async function loadPatch(e, target) {
  try {
    const full = await api(eventUrl(e));
    const patch = full?.part?.state?.metadata?.filediff?.patch || full?.part?.state?.metadata?.diff;
    clear(target).append(patch ? renderDiff(patch) : el('div', { class: 'faint small', text: 'no patch recorded' }));
  } catch (err) {
    clear(target).append(el('div', { class: 'err-text small', text: String(err.message || err) }));
  }
}

function viewOverview() {
  const d = state.data, r = d.record || {}, t = d.totals || {};
  const box = el('div');
  box.append(kv([
    ['build id', el('span', { class: 'mono', text: d.build_id })],
    ['idea', r.idea_id], ['model', r.model], ['created', shortDate(r.created_at)],
    ['structure', `${r.structure} · ${r.n_agents} agent(s) · collab ${r.collab}`],
    ['roles', (r.roles || []).join(', ') || '(decided at runtime)'],
    ['rounds', `${r.rounds_run} run of max ${r.max_rounds} (min ${r.min_rounds})`],
    ['turns', `${r.turns_spent} of max ${r.max_turns}`],
    ['shipped early', truthy(r.shipped_early)], ['qa verified', truthy(r.qa_verified)],
    ['harness ok', truthy(r.harness_ok)], ['status', r.status],
    ['error', r.error ? el('span', { class: 'err-text', text: r.error }) : null],
    ['brief fingerprint', r.brief_fingerprint],
    ['shipped ref', r.shipped_ref],
    ['tool calls', `${num(t.tools)} (${t.tool_errors} failed)`],
    ['cost', usd(t.cost)],
    ['tokens', `${num(t.tokens_total)} total · ${num(t.tokens_output)} generated · ${num(t.tokens_reasoning)} reasoning`],
  ]));
  if (d.manifest?.title) {
    box.append(fold('App manifest (viralbench.json)', el('div', {}, [
      kv([['title', d.manifest.title], ['type', d.manifest.app_type],
        ['summary', d.manifest.summary], ['run', d.manifest.run?.command],
        ['url', d.manifest.run?.url]]),
      (d.manifest.test?.manual || []).length
        ? fold('Manual test steps the build declared', el('ol', { style: 'font-size:11px;padding-left:20px' },
            d.manifest.test.manual.map((s) => el('li', { text: s }))))
        : null,
    ].filter(Boolean)), true));
  }
  return box;
}

function viewThinking() {
  const d = state.data;
  const tr = d.trace || {};
  const cap = tr.summary || {};
  const box = el('div');

  if (cap.source === 'transcript') {
    box.append(el('div', { class: 'note', style: 'margin:12px' }, [
      el('b', {}, ['No thinking was recorded for this build. ']),
      'It predates the capture: opencode only prints reasoning when run with ',
      el('code', { text: '--thinking' }), ', and only the session store holds the ',
      'prompts, the subagents and the file patches. This build has neither, so what ',
      'follows is the assistant\'s written narration and nothing else. Re-run the ',
      'brief to get a full trace.',
    ]));
  } else {
    box.append(el('div', { class: 'note', style: 'margin:12px;border-left-color:var(--purple);background:#1a1626;color:#d6bcff' }, [
      el('b', {}, ['This is the model\'s actual chain of thought, ']),
      'read from opencode\'s session store. ',
      cap.redacted
        ? `${cap.redacted} block(s) came back encrypted by the provider, shown as "redacted", which means it thought and the text is unreadable, not that it did not think. `
        : '',
      'Reasoning is measured in characters here, not tokens: Vertex Anthropic reports ',
      el('code', { text: 'tokens_reasoning: 0' }),
      ' even while returning thousands of characters, so a token rollup would read as "did not think".',
    ]));
  }

  box.append(el('div', { class: 'section', style: 'background:transparent;border:none' }, [
    el('div', { class: 'stats' }, [
      stat(num(cap.stream_chars), 'reasoning chars'),
      stat(num(d.totals.reasoning), 'thoughts'),
      stat(num(cap.redacted), 'redacted'),
      stat(num(cap.prompts), 'prompts'),
      stat(num(cap.sessions), 'sessions'),
      stat(num(d.totals.tokens_reasoning), 'reasoning tokens'),
    ]),
  ]));

  // On a build that delegated, most of the thinking happened below the root
  // session and never reached stdout. That split is the interesting number, not
  // an inconsistency -- so it is shown as a split.
  if (cap.team_chars > 0) {
    const pct = Math.round((cap.team_chars / Math.max(cap.stream_chars, 1)) * 100);
    box.append(el('div', { class: 'note', style: 'margin:0 12px 12px;border-left-color:var(--purple);background:#1a1626;color:#d6bcff' }, [
      el('b', {}, ['Most of this thinking was the team\'s. ']),
      `The orchestrator itself thought ${cap.harness_root.toLocaleString()} characters; `,
      `the subagents it spawned thought ${cap.team_chars.toLocaleString()} more, ${pct}% of the total. `,
      'opencode\'s stdout printer drops every event below the root session, so that share exists only because the harness dumps the session store.',
    ]));
  }

  if (cap.disagrees) {
    box.append(el('div', { class: 'note', style: 'margin:0 12px 12px' }, [
      el('b', {}, ['Two readings of the same file disagree. ']),
      `The harness recorded ${cap.harness_all.toLocaleString()} reasoning characters at build time and `,
      `this viewer reads ${cap.stream_chars.toLocaleString()} from the same session dumps, a gap of `,
      `${Math.abs(cap.harness_all - cap.stream_chars).toLocaleString()}. `,
      'They should agree exactly. Both are shown rather than averaged, because one of the two readers is wrong.',
    ]));
  }

  // Per-turn, because "which turn did it think hardest in" is the question the
  // swimlane cannot answer.
  const perTurn = d.phases.filter((p) => p.reasoning_chars || p.reasoning_parts);
  if (perTurn.length) {
    const max = Math.max(...perTurn.map((p) => p.reasoning_chars || 0), 1);
    box.append(el('table', { class: 'grid' }, [
      el('thead', {}, [el('tr', {}, ['turn', 'thoughts', 'chars', ''].map((h) => el('th', { text: h })))]),
      el('tbody', {}, perTurn.map((p) => el('tr', {
        class: 'clickable',
        onclick: () => {
          const first = state.events.find((e) => e.phase === p.phase && e.kind === 'reasoning');
          if (first) { playback.pause(); playback.seek(first.t); select(first.i); }
        },
      }, [
        el('td', { text: p.phase }),
        el('td', { text: String(p.reasoning_parts) }),
        el('td', { text: num(p.reasoning_chars) }),
        el('td', { style: 'width:45%' }, [el('span', { class: 'bar', style: `width:${(p.reasoning_chars / max) * 100}%;background:var(--purple)` })]),
      ]))),
    ]));
  }

  const thoughts = state.events.filter((e) => e.kind === 'reasoning');
  if (thoughts.length) {
    box.append(el('h2', { style: 'padding:10px 12px 0' }, [
      `Every thought, in order (${thoughts.length})`,
      el('span', { class: 'count' }, [' · click one to jump to it in the run']),
    ]));
    for (const e of thoughts) {
      const laneIndex = d.lanes.findIndex((l) => l.key === e.lane);
      const lane = d.lanes[laneIndex];
      box.append(fold(el('span', {}, [
        el('span', { class: 'lane-dot', style: `background:${laneColor(Math.max(laneIndex, 0))};display:inline-block;margin-right:6px` }),
        el('span', { text: `${e.phase} · ${clockOf(e.t, d.span?.start_ms)}` }),
        lane?.kind === 'subagent' ? el('span', { class: 'badge g-task', style: 'margin-left:6px', text: lane.label }) : null,
        e.redacted
          ? el('span', { class: 'badge', style: 'margin-left:6px;color:var(--warn)', text: 'redacted' })
          : el('span', { class: 'badge', style: 'margin-left:6px', text: `${num(e.chars)} chars` }),
      ].filter(Boolean)), el('div', {}, [
        e.redacted
          ? el('div', { class: 'faint small' }, ['The provider encrypted this thought. It is evidence the model reasoned, not text that can be read.'])
          : el('pre', { class: 'block tall', style: 'color:#d6bcff', text: e.text }),
        el('button', { style: 'margin-top:8px', onclick: () => { playback.pause(); playback.seek(e.t); select(e.i); } }, ['Jump to this moment']),
      ])));
    }
  }

  const texts = state.events.filter((e) => e.kind === 'text');
  if (texts.length) {
    box.append(el('h2', { style: 'padding:10px 12px 0' }, [
      `Narration it wrote for its teammates (${texts.length})`,
    ]));
    for (const e of texts) {
      box.append(fold(`${e.phase} · ${clockOf(e.t, d.span?.start_ms)} · ${num(e.chars)} chars`,
        el('pre', { class: 'block tall', text: e.text })));
    }
  }
  return box;
}

function viewPrompts() {
  const d = state.data;
  const prompts = state.events.filter((e) => e.kind === 'prompt');
  const box = el('div');
  box.append(el('div', { class: 'note', style: 'margin:12px;border-left-color:var(--cyan);background:#122726;color:#9be5e2' }, [
    el('b', {}, ['The prompts, not the answers. ']),
    'Prompts go to opencode on stdin and are never echoed back as events, so they exist ',
    'only in the session store. Without them a trajectory records half a conversation.',
  ]));
  if (!prompts.length) {
    box.append(empty(
      (d.trace?.source === 'transcript')
        ? 'No prompts recorded. This build predates the session-store dump.'
        : 'No prompts found in this build\'s trace.'));
    return box;
  }
  for (const e of prompts) {
    const turn = d.phases.find((p) => p.phase === e.phase);
    box.append(fold(el('span', {}, [
      el('b', { text: e.phase }),
      el('span', { class: 'faint', text: `  ${turn?.role || ''} · ${clockOf(e.t, d.span?.start_ms)} · ${num(e.chars)} chars` }),
    ]), el('div', {}, [
      el('pre', { class: 'block tall', style: 'color:#9be5e2', text: e.text }),
      e.truncated ? loadFull(e) : null,
      el('button', { style: 'margin-top:8px', onclick: () => { playback.pause(); playback.seek(e.t); select(e.i); } }, ['Jump to this moment']),
    ].filter(Boolean)), prompts.length <= 3));
  }
  return box;
}

function viewSessions() {
  const d = state.data;
  const sessions = d.trace?.sessions || [];
  const box = el('div');
  if (!sessions.length) {
    box.append(empty('No session tree. This build predates the session-store dump.'));
    return box;
  }
  box.append(el('div', { class: 'note', style: 'margin:12px' }, [
    el('b', {}, ['Every opencode session this build opened. ']),
    'Depth 0 is the founder or a specialist; anything deeper is work it delegated. ',
    'opencode\'s stdout printer drops every event below the root, so these rows exist ',
    'only because the harness dumps the store.',
  ]));
  box.append(el('table', { class: 'grid' }, [
    el('thead', {}, [el('tr', {}, ['depth', 'agent', 'title', 'model', 'cost', 'in', 'out', 'reasoning', 'session']
      .map((h) => el('th', { text: h })))]),
    el('tbody', {}, sessions.map((sn) => {
      const tk = sn.tokens || {};
      return el('tr', {
        class: 'clickable',
        onclick: () => { state.laneFilter = sn.depth > 0 ? `s:${sn.session_id}` : null; state.tab = 'event'; renderTabs(); refresh(); },
      }, [
        el('td', { text: '·'.repeat(sn.depth) + String(sn.depth) }),
        el('td', { text: sn.agent || '—' }),
        el('td', { title: sn.title, text: (sn.title || '').slice(0, 46) }),
        el('td', { class: 'faint', text: (sn.model || '').split('/').pop() }),
        el('td', { text: sn.cost != null ? usd(sn.cost) : '—' }),
        el('td', { text: num(tk.input) }),
        el('td', { text: num(tk.output) }),
        el('td', { text: num(tk.reasoning) }),
        el('td', { class: 'faint', text: (sn.session_id || '').slice(0, 14) }),
      ]);
    })),
  ]));
  return box;
}

function viewTurns() {
  const d = state.data;
  const table = el('table', { class: 'grid' }, [
    el('thead', {}, [el('tr', {}, ['#', 'phase', 'role', 'round', 'session', 'rc', 'duration', 'tools', 'cost', 'tokens', 'size']
      .map((h) => el('th', { text: h })))]),
    el('tbody', {}, d.phases.map((p, index) => el('tr', {
      class: 'clickable',
      onclick: () => {
        const first = state.events.find((e) => e.phase === p.phase);
        if (first) { playback.pause(); playback.seek(first.t); }
      },
    }, [
      el('td', { text: String(index + 1) }),
      el('td', { text: p.phase }),
      el('td', { text: p.role || '—' }),
      el('td', { text: String(p.round) }),
      el('td', { class: 'faint', title: p.session_id, text: (p.session_id || '').slice(0, 12) }),
      el('td', { class: p.ok === false ? 'err-text' : '', text: String(p.returncode ?? '—') }),
      el('td', { text: dur(p.duration_s) }),
      el('td', { text: `${p.rollup?.tools ?? 0}${p.rollup?.tool_errors ? ` (${p.rollup.tool_errors}✗)` : ''}` }),
      el('td', { text: usd(p.rollup?.cost) }),
      el('td', { text: num(p.rollup?.tokens_total) }),
      el('td', { class: 'faint', text: bytes(p.transcript_bytes) }),
    ]))),
  ]);
  return el('div', {}, [table]);
}

function viewFiles() {
  const files = Object.entries(state.data.files_touched).sort((a, b) => b[1] - a[1]);
  if (!files.length) return empty('No file edits recorded in this build.');
  const max = files[0][1];
  return el('div', {}, [
    el('h2', { style: 'padding:10px 12px 4px' }, [`Files the agents edited (${files.length})`]),
    el('table', { class: 'grid' }, [
      el('thead', {}, [el('tr', {}, ['file', 'edits', ''].map((h) => el('th', { text: h })))]),
      el('tbody', {}, files.map(([name, count]) => el('tr', {}, [
        el('td', { text: name }),
        el('td', { text: String(count) }),
        el('td', { style: 'width:40%' }, [el('span', { class: 'bar', style: `width:${(count / max) * 100}%` })]),
      ]))),
    ]),
  ]);
}

function viewTools() {
  const tools = Object.entries(state.data.tool_counts).sort((a, b) => b[1] - a[1]);
  if (!tools.length) return empty('No tool calls recorded.');
  const max = tools[0][1];
  const errorsByTool = {};
  for (const e of state.events) {
    if (e.kind === 'tool' && e.status === 'error') errorsByTool[e.tool] = (errorsByTool[e.tool] || 0) + 1;
  }
  return el('div', {}, [
    el('table', { class: 'grid' }, [
      el('thead', {}, [el('tr', {}, ['tool', 'calls', 'failed', ''].map((h) => el('th', { text: h })))]),
      el('tbody', {}, tools.map(([name, count]) => el('tr', {
        class: 'clickable',
        onclick: () => { state.groupFilter = null; state.laneFilter = null; refresh(); },
      }, [
        el('td', { text: name }),
        el('td', { text: String(count) }),
        el('td', { class: errorsByTool[name] ? 'err-text' : 'faint', text: String(errorsByTool[name] || 0) }),
        el('td', { style: 'width:40%' }, [el('span', { class: 'bar', style: `width:${(count / max) * 100}%` })]),
      ]))),
    ]),
  ]);
}

function viewDynamic() {
  const d = state.data, o = d.orchestration || {};
  const box = el('div');
  box.append(el('div', { class: 'section', style: 'background:transparent;border:none' }, [
    el('div', { class: 'stats' }, [
      stat(String(o.subagents_spawned ?? d.spawns.length), 'subagents spawned'),
      stat(String(o.peak_concurrent_subagents ?? '—'), 'peak concurrent'),
      stat(String(o.resumed_subagents ?? 0), 'resumed'),
      stat(String(o.failed_spawns ?? 0), 'failed'),
      stat(String(o.orchestrator_turns ?? d.phases.length), 'orchestrator turns'),
      stat(truthy(o.done_signalled), 'signalled done'),
    ]),
  ]));

  if ((d.authored_agents || []).length) {
    box.append(el('h2', { style: 'padding:6px 12px 0' }, [
      `Agents the founder wrote for itself (${d.authored_agents.length})`,
      el('span', { class: 'count' }, [' (nobody specified these; the orchestrator invented the roles)']),
    ]));
    for (const agent of d.authored_agents) {
      box.append(fold(el('span', {}, [el('b', { text: agent.name }), el('span', { class: 'faint', text: `  ${agent.mode || ''} ${agent.temperature ?? ''}` })]),
        el('div', {}, [
          agent.description && el('div', { class: 'dim small', style: 'margin-bottom:6px', text: agent.description }),
          el('pre', { class: 'block', text: agent.body }),
        ].filter(Boolean))));
    }
  }

  if ((d.spawns || []).length) {
    box.append(el('h2', { style: 'padding:10px 12px 0' }, ['Delegations']));
    for (const spawn of d.spawns) {
      box.append(fold(el('span', {}, [
        el('b', { text: spawn.label }),
        el('span', { class: 'badge', style: 'margin-left:8px', text: spawn.subagent_type || '?' }),
        el('span', { class: `badge ${spawn.status === 'error' ? 'err-text' : ''}`, style: 'margin-left:4px', text: spawn.status }),
        spawn.resumed && el('span', { class: 'badge g-task', style: 'margin-left:4px', text: 'resumed' }),
        spawn.start_ms && spawn.end_ms && el('span', { class: 'badge', style: 'margin-left:4px', text: durMs(spawn.end_ms - spawn.start_ms) }),
      ].filter(Boolean)), el('div', {}, [
        kv([['session', spawn.session_id], ['parent', spawn.parent_session_id],
          ['model', spawn.model], ['prompt size', spawn.prompt_chars ? `${num(spawn.prompt_chars)} chars` : null]]),
        spawn.error && el('pre', { class: 'block err-text', text: spawn.error }),
        el('pre', { class: 'block', text: spawn.prompt }),
      ].filter(Boolean))));
    }
  }

  if (o.session_tree) {
    box.append(fold(`Session tree (${o.session_tree.sessions} sessions, depth ${o.session_tree.max_depth})`,
      el('ul', { style: 'font-size:11px' }, (o.session_tree.titles || []).map((t) => el('li', { text: t })))));
  }
  return box;
}

function viewShots() {
  const d = state.data;
  if (!d.screenshots.length) return empty('No screenshots on disk for this build.');
  return el('div', {}, [
    el('h2', { style: 'padding:10px 12px 4px' }, [`Screenshots the agents took (${d.screenshots.length})`,
      el('span', { class: 'count' }, [' from app/.playwright-mcp/'])]),
    el('div', { class: 'shot-grid' }, d.screenshots.map((name) =>
      el('a', { href: `/api/founder/${encodeURIComponent(d.build_id)}/shot/${encodeURIComponent(name)}`, target: '_blank', title: name }, [
        el('img', { class: 'shot', loading: 'lazy', src: `/api/founder/${encodeURIComponent(d.build_id)}/shot/${encodeURIComponent(name)}` }),
      ]))),
  ]);
}

function viewApp() {
  const d = state.data;
  // The tab body scrolls, so height:100% has nothing to resolve against, and a
  // min-height keeps the app pane usable at any window size.
  const box = el('div', { style: 'display:flex;flex-direction:column;min-height:100%' });
  const bar = el('div', { class: 'section', style: 'background:transparent' });
  box.append(bar);

  const manifest = d.manifest || {};
  if (!manifest.run?.command) {
    bar.append(el('div', { class: 'empty', text: 'This build shipped no runnable manifest, so there is nothing to launch.' }));
    return box;
  }

  const frame = el('div', { style: 'flex:1 1 auto;min-height:70vh;background:#fff' });
  const status = el('div', { class: 'small dim', style: 'margin-top:6px' });

  const launch = el('button', {
    class: 'primary',
    onclick: async () => {
      launch.disabled = true;
      status.textContent = 'materialising a throwaway copy, running setup, starting the app…';
      try {
        const res = await post('/api/app/launch', { build_id: d.build_id });
        if (!res.ok) {
          status.replaceChildren(el('div', { class: 'err-text', text: res.error }),
            res.log ? el('pre', { class: 'block', text: res.log }) : el('span'));
          launch.disabled = false;
          return;
        }
        state.app = res;
        status.replaceChildren(el('span', {}, [
          `running at `, el('a', { href: res.url, target: '_blank', text: res.url }),
          ` · ${res.command} · `, el('span', { class: 'faint', text: res.app_dir }),
        ]));
        clear(frame).append(el('iframe', { class: 'appframe', src: res.url }));
        launch.textContent = 'Restart';
        launch.disabled = false;
        stopBtn.disabled = false;
      } catch (err) {
        status.replaceChildren(el('span', { class: 'err-text', text: String(err.message || err) }));
        launch.disabled = false;
      }
    },
  }, ['▶ Launch this app']);

  const stopBtn = el('button', {
    disabled: true,
    onclick: async () => {
      if (!state.app) return;
      await post('/api/app/stop', { session_id: state.app.session_id }).catch(() => {});
      state.app = null;
      clear(frame);
      status.textContent = 'stopped';
      stopBtn.disabled = true;
      launch.textContent = '▶ Launch this app';
    },
  }, ['■ Stop']);

  bar.append(
    el('h2', {}, ['Run the app this build produced']),
    el('div', { class: 'chips' }, [launch, stopBtn,
      chip(manifest.app_type || '?'), chip(manifest.run.command)]),
    el('div', { class: 'small dim', style: 'margin-top:6px' }, [
      'Runs from a throwaway copy under viz/cache, on a free port, behind a proxy that forbids ',
      'browser caching, since otherwise the previous build\'s JavaScript gets reused against this one\'s markup ',
      'and a working app looks broken.',
    ]),
    status,
  );
  box.append(frame);
  return box;
}

function viewPaths() {
  const d = state.data;
  return el('div', {}, [
    el('div', { class: 'note', style: 'margin:12px' }, [
      el('b', {}, ['Everything is already on disk. ']),
      'The raw JSON trajectories live in the build folder below, one file per turn. ',
      'The Download buttons in the toolbar bundle them into a single JSON (or a zip with the screenshots ',
      'and authored agents alongside).',
    ]),
    kv([
      ['build folder', el('span', { class: 'mono', text: d.paths.root })],
      ['build record', el('span', { class: 'mono', text: d.paths.build_json })],
      ['transcripts', el('span', { class: 'mono', text: d.paths.transcript_dir })],
      ['app', el('span', { class: 'mono', text: d.paths.app_dir })],
    ]),
    el('h2', { style: 'padding:6px 12px 0' }, ['One JSON per turn']),
    el('table', { class: 'grid' }, [
      el('thead', {}, [el('tr', {}, ['file', 'bytes', 'events'].map((h) => el('th', { text: h })))]),
      el('tbody', {}, d.phases.map((p) => el('tr', {}, [
        el('td', { text: `${p.phase}.json` }),
        el('td', { text: bytes(p.transcript_bytes) }),
        el('td', { text: String((p.rollup?.tools || 0) + (p.rollup?.texts || 0) + (p.rollup?.steps || 0)) }),
      ]))),
    ]),
    (d.crowd_runs || []).length ? el('div', {}, [
      el('h2', { style: 'padding:10px 12px 0' }, [`Crowd runs against this build (${d.crowd_runs.length})`]),
      el('table', { class: 'grid' }, [
        el('tbody', {}, d.crowd_runs.slice(0, 40).map((r) => el('tr', { class: 'clickable', onclick: () => { location.href = `/crowd?run=${encodeURIComponent(r.run_id)}`; } }, [
          el('td', { text: r.run_id.replace(`${d.build_id}__`, '') }),
          el('td', { text: r.app_type }),
          el('td', { text: `${r.posts}p ${r.likes}♥ ${r.comments}💬` }),
          el('td', { text: r.ok ? 'ok' : 'failed', class: r.ok ? 'ok-text' : 'err-text' }),
        ]))),
      ]),
    ]) : null,
  ].filter(Boolean));
}

// ---------------------------------------------------------------- build browser

let browseRows = [];

async function openBrowser() {
  $('#browser').hidden = false;
  $('#searchInput').focus();
  await refreshBrowse();
}

async function refreshBrowse() {
  const query = new URLSearchParams();
  const search = $('#searchInput').value.trim();
  if (search) query.set('q', search);
  if ($('#modeFilter').value) query.set('mode', $('#modeFilter').value);
  if ($('#statusFilter').value) query.set('status', $('#statusFilter').value);
  if ($('#crowdOnly').checked) query.set('crowd', '1');
  query.set('limit', '500');

  const body = $('#browseBody');
  body.replaceChildren(el('div', { class: 'empty' }, [el('span', { class: 'spin', text: '◐' }), ' searching…']));
  try {
    const result = await api(`/api/builds?${query}`);
    browseRows = result.builds;
    if (!$('#modeFilter').options.length) {
      $('#modeFilter').append(el('option', { value: '', text: 'all modes' }),
        ...Object.entries(result.modes).sort((a, b) => b[1] - a[1])
          .map(([m, c]) => el('option', { value: m, text: `${m} (${c})` })));
      $('#statusFilter').append(el('option', { value: '', text: 'any status' }),
        ...['ok', 'harness_failed', 'harness_timeout', 'manifest_missing', 'manifest_invalid']
          .map((s) => el('option', { value: s, text: s })));
    }
    body.replaceChildren(el('div', { class: 'small dim', style: 'padding:6px 14px' },
      [`${result.matched} of ${result.total} builds${result.matched > browseRows.length ? ` (showing ${browseRows.length})` : ''}`]),
      el('table', { class: 'grid' }, [
        el('thead', {}, [el('tr', {}, ['idea', 'mode', 'model', 'status', 'turns', 'time', 'transcripts', 'crowd', 'created', 'build id']
          .map((h) => el('th', { text: h })))]),
        el('tbody', {}, browseRows.map((row) => el('tr', {
          class: 'clickable',
          onclick: () => { $('#browser').hidden = true; load(row.build_id); },
        }, [
          el('td', { text: row.idea_id }),
          el('td', {}, [chip(row.mode, { solo: 'accent', team: 'ok', dynamic: 'warn' }[row.mode] || '')]),
          el('td', { text: row.model }),
          el('td', { class: row.status === 'ok' ? 'ok-text' : 'err-text', text: row.status }),
          el('td', { text: String(row.turns_spent ?? '—') }),
          el('td', { text: dur(row.duration_s) }),
          el('td', { class: 'faint', text: bytes(row.transcript_bytes) }),
          el('td', { text: row.crowd_runs ? String(row.crowd_runs) : '—' }),
          el('td', { class: 'faint', text: shortDate(row.created_at) }),
          el('td', { class: 'faint', text: row.build_id }),
        ]))),
      ]));
  } catch (err) {
    body.replaceChildren(el('div', { class: 'empty err-text', text: String(err.message || err) }));
  }
}

// ---------------------------------------------------------------- wiring

$('#loadBtn').addEventListener('click', () => load($('#buildInput').value.trim()));
$('#buildInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') load($('#buildInput').value.trim()); });
$('#browseBtn').addEventListener('click', openBrowser);
$('#closeBrowse').addEventListener('click', () => { $('#browser').hidden = true; });
$('#browser').addEventListener('click', (e) => { if (e.target.id === 'browser') $('#browser').hidden = true; });
$('#crowdBtn').addEventListener('click', () => { location.href = `/crowd?build=${encodeURIComponent(state.data.build_id)}`; });
$('#dlJson').addEventListener('click', () => { location.href = `/api/founder/${encodeURIComponent(state.data.build_id)}/download`; });
$('#dlZip').addEventListener('click', () => { location.href = `/api/founder/${encodeURIComponent(state.data.build_id)}/download?format=zip`; });

let searchTimer = null;
for (const id of ['#searchInput', '#modeFilter', '#statusFilter', '#crowdOnly']) {
  $(id).addEventListener(id === '#searchInput' ? 'input' : 'change', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(refreshBrowse, id === '#searchInput' ? 220 : 0);
  });
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
    if (e.key === 'Escape') $('#browser').hidden = true;
    return;
  }
  if (e.key === ' ') { e.preventDefault(); playback.toggle(); }
  else if (e.key === 'ArrowRight') step(1);
  else if (e.key === 'ArrowLeft') step(-1);
  else if (e.key === 'Escape') $('#browser').hidden = true;
  else if (e.key === '/') { e.preventDefault(); openBrowser(); }
});

const initial = params.get('build');
if (initial) load(initial);
else openBrowser();
