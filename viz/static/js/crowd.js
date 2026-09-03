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

/* Crowd trajectory viewer.
 *
 * A crowd run is two recordings on two different clocks, and the UI keeps both.
 *
 * The *social* clock is a small integer: 0 is sign-ups and the seeded follow
 * graph, 1 is the founder's launch post, round N happens at N+1, and the last
 * step is the exit interview. Playback walks it, and the graph, the feed and the
 * action log are all projections of where it has got to.
 *
 * The *trial* clock is real wall-clock time inside one agent's hands-on session
 * with the app -- every click and keystroke, with the page state that came back
 * and the screenshots it took. That plays separately, because it happens "inside"
 * a single tick of the social clock.
 */

import {
  $, agentColor, api, bytes, chip, clear, dur, el, empty, fold, initials, kv,
  num, params, Playback, post, stat, toast, truthy,
} from './common.js';

const state = {
  run: null,
  agents: [],
  byId: new Map(),
  step: 0,
  selectedAgent: null,
  hoverAgent: null,
  selectedPost: null,
  trial: null,
  trialIndex: 0,
  tab: 'agent',
  midTab: 'feed',
  app: null,
  layout: [],
};

const playback = new Playback({ onTick: onClock, onState: renderTransport });
const trialPlayback = new Playback({ onTick: onTrialClock, onState: () => renderTabBody() });

const TIER_COLORS = { founder: '#ffd166', trier: '#5aa9ff', latecomer: '#ff7fc4', reactor: '#6b7488' };

// ---------------------------------------------------------------- loading

async function load(id) {
  if (!id) return;
  $('#midBody').replaceChildren(el('div', { class: 'empty' }, [el('span', { class: 'spin', text: '◐' }), ' loading…']));
  // A build id and a run id are both plausible things to paste. If it is a build,
  // show its runs rather than an error.
  try {
    const run = await api(`/api/crowd/${encodeURIComponent(id)}`);
    adopt(run);
  } catch {
    try {
      const listed = await api(`/api/crowd/runs?build_id=${encodeURIComponent(id)}&limit=200`);
      if (listed.runs.length === 1) return load(listed.runs[0].run_id);
      if (listed.runs.length) { openBrowser(id); return; }
      toast(`No crowd run or build matches "${id}"`, true);
    } catch (err) { toast(String(err.message || err), true); }
    $('#midBody').replaceChildren(empty('Nothing loaded.'));
  }
}

function adopt(run) {
  state.run = run;
  state.agents = run.agents;
  state.byId = new Map(run.agents.map((a) => [a.id, a]));
  // Open on the agent with the deepest trial: that is the replay worth watching,
  // and landing on agent 1 usually means landing on a five-step skim.
  const withTrials = run.agents.filter((a) => a.has_trace);
  withTrials.sort((a, b) => (b.trial?.n_steps || 0) - (a.trial?.n_steps || 0));
  state.selectedAgent = withTrials[0]?.id ?? run.agents[0]?.id ?? null;
  state.selectedPost = null;
  state.trial = null;
  state.tab = 'agent';
  state.midTab = 'feed';
  params.set('run', run.run_id);
  $('#runInput').value = run.run_id;
  for (const id of ['dlJson', 'dlZip', 'rescueBtn']) $(`#${id}`).disabled = false;
  $('#founderBtn').disabled = !run.build_id;

  layoutGraph();
  renderHead();
  renderAgents();
  renderMidTabs();
  renderTabs();

  const last = run.timeline.length ? run.timeline[run.timeline.length - 1].t : 1;
  playback.setRange(0, last, 0.7);   // ~1.4 seconds per simulation step
  playback.seek(last);
  onClock(last);
  loadTrial(state.selectedAgent);
}

// ---------------------------------------------------------------- header

const pct = (v) => (v != null ? `${Math.round(v * 100)}%` : '—');
const mean = (v) => (v != null ? v.toFixed(1) : '—');

/** Spell out which of the two verdict passes a headline number came from.
 *
 *  A crowd agent is scored twice and the passes disagree: the TRIAL verdict is
 *  written by the agents who drove the app, the INTERVIEW is asked of
 *  everyone at the end, including agents who only ever saw a post about it. */
function verdictTip(what, trial, interview) {
  const t = state.run.verdicts?.triers || {};
  const i = state.run.verdicts?.interviews || {};
  return [
    `${what}: from the hands-on TRIAL verdicts.`,
    '',
    `trial      ${trial}   (${t.n ?? 0} agents who opened and used the app)`,
    `interview  ${interview}   (${i.n ?? 0} agents asked at the end, app-users or not)`,
    '',
    'Click for the full breakdown in the Verdicts tab.',
  ].join('\n');
}

function showVerdicts() {
  state.midTab = 'verdicts';
  renderMidTabs();
  $('#midTabs')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderHead() {
  const r = state.run;
  const cfg = r.config || {};
  const e = r.engagement || {};
  const triers = r.verdicts?.triers || {};
  const interviews = r.verdicts?.interviews || {};

  const f = r.founder || {};
  const armClass = { solo: 'accent', team: 'ok', dynamic: 'warn' }[f.arm] || '';

  // Which pipeline and model BUILT the app, first -- a crowd score is meaningless
  // without its subject. Kept visually distinct from the crowd's own settings,
  // because the model in a run's config is the crowd's, not the founder's, and
  // reading one as the other is the easy mistake here.
  const founderChips = f.on_disk
    ? [
        el('span', {
          class: `chip mode ${armClass}`,
          style: 'cursor:pointer',
          title: `Open this build in the founder trajectory viewer\n${f.build_id}`,
          onclick: () => { location.href = `/founder?build=${encodeURIComponent(f.build_id)}`; },
        }, [`built by: ${f.arm_label}`]),
        el('span', {
          class: 'chip accent',
          style: 'cursor:pointer',
          title: 'the founder model, the one under test',
          onclick: () => { location.href = `/founder?build=${encodeURIComponent(f.build_id)}`; },
        }, [f.model_short || 'unknown model']),
        f.status && f.status !== 'ok' && chip(`build ${f.status}`, 'err'),
        f.turns_spent != null
          && chip(`${f.turns_spent} founder turn${f.turns_spent === 1 ? '' : 's'}`),
        f.subagents_spawned > 0
          && chip(`⑂ ${f.subagents_spawned} subagents`, 'purple'),
        f.arm === 'dynamic' && !f.subagents_spawned
          && chip('⑂ no subagents, it chose to work alone', 'warn'),
        f.reasoning_chars > 0 && chip(`🧠 ${num(f.reasoning_chars)} chars thinking`, 'purple'),
        f.qa_verified === true && chip('QA verified', 'ok'),
      ]
    : [chip('founder build no longer on disk', 'warn')];

  const chips = [
    ...founderChips,
    el('span', { class: 'faint', style: 'margin:0 2px' }, ['|']),
    chip(r.app_type || 'unknown app', 'accent'),
    chip(r.ok ? 'healthy' : 'unhealthy', r.ok ? 'ok' : 'err'),
    chip(`arch v${r.arch_version}`),
    chip(`${cfg.n_agents ?? r.agents.length - 1} agents`),
    chip(`${r.rounds ? r.rounds.length : '?'} rounds`),
    // Labelled: this is the CROWD's model, not the founder's.
    chip(`crowd: ${cfg.model_id || '?'}`),
    chip(`recsys ${cfg.recsys_type || '?'}`),
    cfg.seed != null && chip(`seed ${cfg.seed}`),
    r.undeliverable && chip('undeliverable build', 'err'),
    r.validity?.does_what_it_claims === false && chip('validity gate: fails its claim', 'err'),
    e.exposure && chip(`${e.exposure.distinct_feeds}/${e.exposure.agents_with_feed} distinct feeds`,
      e.exposure.uniform ? 'warn' : 'ok'),
  ].filter(Boolean);

  clear($('#head')).append(el('div', { style: 'display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap' }, [
    el('div', { style: 'flex:1 1 340px;min-width:0' }, [
      el('div', { style: 'display:flex;gap:9px;align-items:baseline;flex-wrap:wrap' }, [
        el('div', { style: 'font-size:16px;font-weight:700' }, [
          f.app_title || r.build_id.split('__')[0],
        ]),
        el('div', { class: 'dim small' }, [r.build_id.split('__')[0]]),
        el('div', { class: 'mono faint small', text: r.run_id }),
      ]),
      el('div', { class: 'chips', style: 'margin-top:6px' }, chips),
    ]),
    el('div', { class: 'stats' }, [
      stat(num(e.posts), 'posts'),
      stat(num(e.likes), 'likes'),
      stat(num(e.comments), 'comments'),
      stat(num(e.reposts), 'reposts'),
      stat(num(e.follows), 'follows'),
      // Both of these come from the hands-on TRIAL verdicts, not the exit
      // interview -- two different numbers that a bare "delight" label invites
      // you to confuse. The label names the source and the tooltip shows the
      // other figure, so the reader can see whether they agree.
      stat(pct(triers.would_use_rate), 'would use (trial)', {
        title: verdictTip('would use', pct(triers.would_use_rate), pct(interviews.would_use_rate)),
        onclick: showVerdicts,
      }),
      stat(mean(triers.delight_mean), 'delight (trial)', {
        title: verdictTip('delight (0–10)', mean(triers.delight_mean), mean(interviews.delight_mean)),
        onclick: showVerdicts,
      }),
      stat(dur(r.duration_s), 'wall clock'),
    ]),
  ]));
}

// ---------------------------------------------------------------- graph

/** Spring layout, settled once on load.
 *
 *  Continuous animation would be prettier and would also mean the node you were
 *  looking at drifts away while you read it. The layout is computed once so a
 *  position means something stable for the whole session. */
function layoutGraph() {
  const nodes = state.agents.map((a) => ({ id: a.id, x: 0, y: 0, vx: 0, vy: 0 }));
  const index = new Map(nodes.map((n, i) => [n.id, i]));
  const edges = state.run.follows
    .map((f) => [index.get(f.source), index.get(f.target)])
    .filter(([a, b]) => a !== undefined && b !== undefined);

  const n = nodes.length || 1;
  nodes.forEach((node, i) => {
    // The founder is the origin of everything, so anchoring it at the centre makes
    // the picture read as "how far did this travel from the source".
    if (node.id === 0) { node.x = 0; node.y = 0; return; }
    const angle = (i / n) * Math.PI * 2;
    node.x = Math.cos(angle) * 160;
    node.y = Math.sin(angle) * 160;
  });

  // Everyone follows the founder plus a few interest-neighbours, so the graph is
  // dense enough to collapse into a hairball at the textbook constant. A larger
  // ideal edge length and weaker attraction trade compactness for being able to
  // tell the nodes apart, which is the entire job of this picture.
  const k = 150;
  const ITERATIONS = 420;
  for (let iteration = 0; iteration < ITERATIONS; iteration += 1) {
    const cool = 1 - iteration / ITERATIONS;
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        let dx = nodes[i].x - nodes[j].x;
        let dy = nodes[i].y - nodes[j].y;
        let distance = Math.hypot(dx, dy) || 0.01;
        if (distance < 1) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; distance = 1; }
        const force = (k * k) / (distance * distance);
        const fx = (dx / distance) * force, fy = (dy / distance) * force;
        nodes[i].vx += fx; nodes[i].vy += fy;
        nodes[j].vx -= fx; nodes[j].vy -= fy;
      }
    }
    for (const [a, b] of edges) {
      const dx = nodes[a].x - nodes[b].x, dy = nodes[a].y - nodes[b].y;
      const distance = Math.hypot(dx, dy) || 0.01;
      const force = (distance * distance) / k / 14;
      const fx = (dx / distance) * force, fy = (dy / distance) * force;
      nodes[a].vx -= fx; nodes[a].vy -= fy;
      nodes[b].vx += fx; nodes[b].vy += fy;
    }
    for (const node of nodes) {
      node.vx -= node.x * 0.02;
      node.vy -= node.y * 0.02;
      if (node.id === 0) { node.vx = node.vy = 0; continue; }
      const speed = Math.hypot(node.vx, node.vy) || 1;
      const capped = Math.min(speed, 30 * cool);
      node.x += (node.vx / speed) * capped;
      node.y += (node.vy / speed) * capped;
      node.vx *= 0.4; node.vy *= 0.4;
    }
  }
  // Recentre on the centroid so the founder does not end up flush against an edge
  // only because the crowd settled asymmetrically around it.
  const cx = nodes.reduce((sum, node) => sum + node.x, 0) / n;
  const cy = nodes.reduce((sum, node) => sum + node.y, 0) / n;
  for (const node of nodes) { node.x -= cx; node.y -= cy; }
  state.layout = nodes;
}

function drawGraph() {
  const canvas = $('#graph');
  const wrap = canvas.parentElement;
  const width = wrap.clientWidth || 380;
  const height = 380;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!state.layout.length) return;

  const xs = state.layout.map((n) => n.x), ys = state.layout.map((n) => n.y);
  const pad = 26;
  const spanX = Math.max(...xs) - Math.min(...xs) || 1;
  const spanY = Math.max(...ys) - Math.min(...ys) || 1;
  const scale = Math.min((width - pad * 2) / spanX, (height - pad * 2) / spanY);
  const minX = Math.min(...xs), minY = Math.min(...ys);
  const px = (x) => pad + (x - minX) * scale;
  const py = (y) => pad + (y - minY) * scale;
  const pos = new Map(state.layout.map((n) => [n.id, { x: px(n.x), y: py(n.y) }]));

  const t = state.step;

  // Engagement received so far, so a node grows as the crowd notices it.
  const received = new Map();
  for (const post of state.run.posts) {
    if (post.t > t) continue;
    received.set(post.user_id, (received.get(post.user_id) || 0) + 1 + post.likes + post.shares);
  }

  ctx.lineWidth = 1;
  for (const follow of state.run.follows) {
    const a = pos.get(follow.source), b = pos.get(follow.target);
    if (!a || !b || follow.t > t) continue;
    ctx.strokeStyle = 'rgba(90,110,150,0.16)';
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }

  // Amplification at exactly this step: who is passing whose post along right now.
  const authorOf = new Map(state.run.posts.map((p) => [p.post_id, p.user_id]));
  for (const action of state.run.actions) {
    if (action.t !== t) continue;
    const source = action.source_post ? authorOf.get(action.source_post) : null;
    if (source == null) continue;
    const from = pos.get(action.user_id), to = pos.get(source);
    if (!from || !to) continue;
    ctx.strokeStyle = action.action === 'quote_post' ? 'rgba(255,127,196,.85)' : 'rgba(78,201,138,.8)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const mx = (from.x + to.x) / 2 + (to.y - from.y) * 0.16;
    const my = (from.y + to.y) / 2 - (to.x - from.x) * 0.16;
    ctx.moveTo(from.x, from.y);
    ctx.quadraticCurveTo(mx, my, to.x, to.y);
    ctx.stroke();
  }
  ctx.lineWidth = 1;

  const actingNow = new Set(
    state.run.actions.filter((a) => a.t === t && a.action !== 'refresh' && a.action !== 'do_nothing')
      .map((a) => a.user_id),
  );

  for (const agent of state.agents) {
    const point = pos.get(agent.id);
    if (!point) continue;
    const weight = received.get(agent.id) || 0;
    const radius = agent.id === 0 ? 11 : 4.5 + Math.min(Math.sqrt(weight) * 1.5, 8);
    const color = TIER_COLORS[agent.tier] || '#6b7488';

    if (actingNow.has(agent.id)) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius + 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,.10)';
      ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = agent.id === 0 || weight > 0 ? 1 : 0.5;
    ctx.fill();
    ctx.globalAlpha = 1;
    if (state.selectedAgent === agent.id) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(point.x, point.y, radius + 3.5, 0, Math.PI * 2); ctx.stroke();
      ctx.lineWidth = 1;
    }
  }

  ctx.font = '600 10px ui-monospace, monospace';
  ctx.textAlign = 'center';
  const founder = pos.get(0);
  if (founder) {
    ctx.fillStyle = '#ffd166';
    ctx.fillText('@founder', founder.x, founder.y - 16);
  }
  // Label whoever the crowd is reacting to, plus whatever is selected or
  // hovered. Labelling every node is unreadable, and labelling none makes the
  // graph mute.
  const loudest = [...received.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([id]) => id);
  for (const id of new Set([...loudest, state.selectedAgent, state.hoverAgent])) {
    if (id == null || id === 0) continue;
    const point = pos.get(id);
    const agent = state.byId.get(id);
    if (!point || !agent) continue;
    ctx.fillStyle = id === state.hoverAgent || id === state.selectedAgent ? '#ffffff' : 'rgba(215,221,237,.65)';
    ctx.fillText(`@${agent.username}`, point.x, point.y - 13);
  }
  canvas._pos = pos;
}

function graphHitTest(event) {
  const canvas = $('#graph');
  const pos = canvas._pos;
  if (!pos) return null;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  let best = null, bestDistance = 16;
  for (const [id, point] of pos) {
    const distance = Math.hypot(point.x - x, point.y - y);
    if (distance < bestDistance) { best = id; bestDistance = distance; }
  }
  return best;
}

//: Actions that are the crowd looking rather than doing. Counting them alongside
//: posts and reposts is what makes a round where nobody engaged look busy.
const PASSIVE_ACTIONS = new Set(['refresh', 'do_nothing']);

/** Split a step's action counts into what the crowd did vs merely looked at. */
function actionSplit(counts) {
  let acted = 0;
  let passive = 0;
  for (const [name, n] of Object.entries(counts || {})) {
    if (PASSIVE_ACTIONS.has(name)) passive += n; else acted += n;
  }
  return { acted, passive, total: acted + passive };
}

function stepTooltip(step, split) {
  const listed = Object.entries(step.counts || {})
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `  ${k} ${v}`)
    .join('\n');
  return [
    `${step.label} (timestep ${step.t})`,
    `${split.acted} action${split.acted === 1 ? '' : 's'} by ${step.actors}`
      + ` agent${step.actors === 1 ? '' : 's'}`,
    split.passive ? `${split.passive} passive (feed refresh / did nothing)` : null,
    '',
    listed,
  ].filter((line) => line !== null).join('\n');
}

// ---------------------------------------------------------------- transport

function renderTransport() {
  const box = $('#transport');
  const timeline = state.run?.timeline || [];
  if (!box.dataset.built) {
    box.dataset.built = '1';
    clear(box).append(el('div', { class: 'transport' }, [
      el('button', { id: 'playBtn', class: 'primary', onclick: () => playback.toggle() }, ['▶ Play']),
      el('button', { onclick: () => { playback.pause(); playback.seek(0); } }, ['⏮']),
      el('button', { onclick: () => { playback.pause(); playback.seek(Math.floor(playback.value) - 1); } }, ['◀']),
      el('button', { onclick: () => { playback.pause(); playback.seek(Math.floor(playback.value) + 1); } }, ['▶']),
      el('select', { id: 'speedSel', onchange: (e) => playback.setSpeed(Number(e.target.value)) },
        [0.5, 1, 2, 4].map((s) => el('option', { value: String(s), text: `${s}×`, selected: s === 1 }))),
      el('div', { class: 'chips', id: 'stepChips', style: 'flex:1 1 auto' }),
      el('span', {
        class: 'clock',
        id: 'clockLabel',
        title: 'Each pill is one timestep of the simulation clock. The number is how '
          + 'many things the crowd DID at that step: posts, reposts, quotes, comments, '
          + 'likes and follows. Feed refreshes and explicit do-nothings are excluded, '
          + 'since every agent logs one every round. Hover a pill for the full breakdown.',
      }),
    ]));
  }
  const play = $('#playBtn');
  if (play) play.textContent = playback.playing ? '❚❚ Pause' : '▶ Play';

  const chipBox = $('#stepChips');
  if (chipBox && timeline.length) {
    clear(chipBox).append(...timeline.map((s) => {
      const split = actionSplit(s.counts);
      return el('span', {
        class: `chip${s.t === state.step ? ' accent' : ''}`,
        style: `cursor:pointer;opacity:${s.t <= state.step ? 1 : 0.4}`,
        title: stepTooltip(s, split),
        onclick: () => { playback.pause(); playback.seek(s.t); },
      }, [
        `t${s.t} ${s.label}`,
        // The count that means "something happened". The raw total counts a
        // feed refresh and a decision to do nothing as actions, so it barely
        // moves between rounds even when the crowd has stopped engaging.
        el('b', { style: 'margin-left:5px', text: String(split.acted) }),
        el('span', {
          class: 'faint', style: 'margin-left:3px',
          text: split.acted === 1 ? 'act' : 'acts',
        }),
      ]);
    }));
  }
  const label = $('#clockLabel');
  if (label && timeline.length) {
    const now = timeline.find((s) => s.t === state.step);
    const split = now ? actionSplit(now.counts) : null;
    label.textContent = `step ${state.step}/${timeline[timeline.length - 1].t}`
      + (now
        ? ` · ${now.label} · ${split.acted} act${split.acted === 1 ? '' : 's'}`
          + ` by ${now.actors} agent${now.actors === 1 ? '' : 's'}`
        : '');
  }
}

function onClock(value) {
  if (!state.run) return;
  const step = Math.floor(value);
  const changed = step !== state.step;
  state.step = step;
  drawGraph();
  renderTransport();
  if (changed) { renderMidBody(); renderAgents(); }
}

// ---------------------------------------------------------------- roster

function renderAgents() {
  const box = clear($('#agents'));
  $('#agentCount').textContent = `· ${state.agents.length} incl. the founder`;
  $('#graphCount').textContent = `· ${state.run.follows.length} follows, ${state.run.posts.filter((p) => p.t <= state.step).length} posts so far`;

  const posted = new Map();
  for (const post of state.run.posts) {
    if (post.t <= state.step) posted.set(post.user_id, (posted.get(post.user_id) || 0) + 1);
  }

  for (const agent of state.agents) {
    const color = TIER_COLORS[agent.tier] || '#6b7488';
    const verdict = agent.trial || agent.interview || {};
    box.append(el('div', {
      class: `agent-row${state.selectedAgent === agent.id ? ' sel' : ''}`,
      onclick: () => selectAgent(agent.id),
    }, [
      el('div', { class: 'avatar', style: `background:${color}`, text: String(agent.id) }),
      el('div', { style: 'min-width:0' }, [
        el('div', { class: 'mono', style: 'overflow:hidden;text-overflow:ellipsis;white-space:nowrap' }, [
          `@${agent.username}`,
          agent.has_trace ? el('span', { class: 'badge', style: 'margin-left:5px', text: 'trial' }) : null,
        ].filter(Boolean)),
        el('div', { class: 'faint', style: 'font-size:10px' }, [
          `${agent.tier || 'crowd'} · ${agent.followers}f · ${posted.get(agent.id) || 0} posts`,
        ]),
      ]),
      el('div', { class: 'nowrap small', style: `color:${verdict.would_use === true ? 'var(--ok)' : verdict.would_use === false ? 'var(--err)' : 'var(--fg-faint)'}` }, [
        verdict.delight != null ? `♦${verdict.delight}` : '—',
      ]),
    ]));
  }
}

function selectAgent(id) {
  state.selectedAgent = id;
  state.tab = 'agent';
  renderAgents();
  drawGraph();
  renderTabs();
  loadTrial(id);
}

async function loadTrial(id) {
  state.trial = null;
  state.trialIndex = 0;
  const agent = state.byId.get(id);
  if (!agent?.has_trace) { renderTabBody(); return; }
  try {
    state.trial = await api(`/api/crowd/${encodeURIComponent(state.run.run_id)}/trial/${id}`);
    const steps = state.trial.steps;
    if (steps.length) {
      const first = steps[0].ts || 0;
      const last = steps[steps.length - 1].ts || first + 1;
      trialPlayback.setRange(first, Math.max(last + 1, first + 1), Math.max((last - first) / 20, 1));
      trialPlayback.seek(first);
      state.trialIndex = 0;
    }
  } catch (err) { toast(String(err.message || err), true); }
  renderTabBody();
}

function onTrialClock(value) {
  if (!state.trial) return;
  const steps = state.trial.steps;
  let index = 0;
  for (let i = 0; i < steps.length; i += 1) if ((steps[i].ts || 0) <= value) index = i;
  if (index !== state.trialIndex) { state.trialIndex = index; renderTabBody(); }
}

// ---------------------------------------------------------------- middle pane

function renderMidTabs() {
  const tabs = [['feed', 'Feed'], ['actions', 'Action log'], ['verdicts', 'Verdicts'], ['autorating', 'Autorating'], ['health', 'Run health']];
  clear($('#midTabs')).append(...tabs.map(([key, label]) =>
    el('div', { class: `tab${state.midTab === key ? ' active' : ''}`, onclick: () => { state.midTab = key; renderMidTabs(); } }, [label])));
  renderMidBody();
}

function renderMidBody() {
  const body = clear($('#midBody'));
  if (!state.run) return;
  const views = { feed: viewFeed, actions: viewActions, verdicts: viewVerdicts, autorating: viewAutorating, health: viewHealth };
  body.append((views[state.midTab] || viewFeed)());
}

function viewFeed() {
  const posts = state.run.posts.filter((p) => p.t <= state.step);
  if (!posts.length) return empty('No posts yet at this point in the simulation.');
  const box = el('div');
  for (const post of [...posts].reverse()) box.append(postCard(post));
  return box;
}

function postCard(post) {
  const author = state.byId.get(post.user_id);
  const original = post.original_post_id ? state.run.posts.find((p) => p.post_id === post.original_post_id) : null;
  const originalAuthor = original ? state.byId.get(original.user_id) : null;
  const isLaunch = post.post_id === state.run.launch_post_id;

  const card = el('div', {
    class: `post${state.selectedPost === post.post_id ? ' sel' : ''}${isLaunch ? ' launch' : ''}`,
    onclick: () => { state.selectedPost = post.post_id; state.tab = 'post'; renderTabs(); renderMidBody(); },
  }, [
    el('div', { class: 'post-head' }, [
      el('span', { class: 'who', style: `color:${TIER_COLORS[author?.tier] || '#5aa9ff'}`, text: `@${author?.username || post.user_id}` }),
      el('span', { class: 'faint', text: `t${post.t}` }),
      isLaunch && chip('launch post', 'warn'),
      post.kind === 'repost' && chip(`reposted @${originalAuthor?.username || '?'}`, 'ok'),
      post.kind === 'quote' && chip(`quoted @${originalAuthor?.username || '?'}`, 'purple'),
    ].filter(Boolean)),
  ]);

  const text = post.quote_content || post.content;
  if (text) card.append(el('div', { class: 'post-body', text: text.length > 700 ? `${text.slice(0, 700)}…` : text }));
  else if (original) card.append(el('div', { class: 'post-body faint', text: `↻ ${original.content.slice(0, 300)}` }));

  card.append(el('div', { class: 'post-meta' }, [
    `#${post.post_id}`, `♥ ${post.likes}`, `↻ ${post.shares}`,
    `💬 ${post.comments.length}`, post.dislikes ? `👎 ${post.dislikes}` : null,
    post.reports ? `⚑ ${post.reports}` : null,
  ].filter(Boolean)));

  for (const comment of post.comments.filter((c) => c.t <= state.step)) {
    const who = state.byId.get(comment.user_id);
    card.append(el('div', { class: 'comment' }, [
      el('span', { class: 'who', text: `@${who?.username || comment.user_id} ` }),
      comment.content.length > 340 ? `${comment.content.slice(0, 340)}…` : comment.content,
      comment.likes ? el('span', { class: 'faint', text: `  ♥${comment.likes}` }) : null,
    ].filter(Boolean)));
  }
  return card;
}

function viewActions() {
  const actions = state.run.actions.filter((a) => a.t <= state.step);
  if (!actions.length) return empty('Nothing has happened yet.');
  const rows = [...actions].reverse().slice(0, 900);
  return el('table', { class: 'grid' }, [
    el('thead', {}, [el('tr', {}, ['t', 'who', 'action', 'what'].map((h) => el('th', { text: h })))]),
    el('tbody', {}, rows.map((a) => {
      const who = state.byId.get(a.user_id);
      return el('tr', { class: 'clickable', onclick: () => selectAgent(a.user_id) }, [
        el('td', { class: 'faint', text: `t${a.t}` }),
        el('td', { style: `color:${TIER_COLORS[who?.tier] || '#6b7488'}`, text: `@${who?.username || a.user_id}` }),
        el('td', { text: a.action }),
        el('td', { class: 'faint', style: 'max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap' }, [
          a.text || a.query || (a.post_id ? `#${a.post_id}` : '') + (a.source_post ? ` ← #${a.source_post}` : '')
          || (a.n_posts != null ? `${a.n_posts} posts in feed` : '') || (a.response ? a.response.slice(0, 120) : ''),
        ]),
      ]);
    })),
  ]);
}

function viewVerdicts() {
  const v = state.run.verdicts || {};
  const box = el('div');
  const triers = v.triers || {}, interviews = v.interviews || {};

  box.append(el('div', { class: 'section', style: 'background:transparent;border:none' }, [
    el('h2', {}, ['Hands-on triers', el('span', { class: 'count' }, [` · ${triers.n ?? 0} agents who used the app`])]),
    el('div', { class: 'dim small', style: 'padding:0 12px 8px' }, [
      'Written after driving the real app in a browser. These are the numbers shown in the header.',
    ]),
    el('div', { class: 'stats' }, [
      stat(pct(triers.would_use_rate), 'would use'),
      stat(pct(triers.would_share_rate), 'would share'),
      stat(mean(triers.delight_mean), 'delight'),
      stat(mean(triers.craft_mean), 'craft'),
      stat(mean(triers.functionality_mean), 'function'),
      stat(mean(triers.usability_mean), 'usability'),
      stat(mean(triers.design_mean), 'design'),
    ]),
  ]));

  if (triers.delight_histogram) box.append(histogram(triers.delight_histogram, 'delight distribution (0–10)'));

  box.append(el('div', { class: 'section', style: 'background:transparent;border:none' }, [
    el('h2', {}, ['Exit interviews', el('span', { class: 'count' }, [` · ${interviews.n ?? 0} answered of ${interviews.expected ?? '?'}`])]),
    el('div', { class: 'dim small', style: 'padding:0 12px 8px' }, [
      'Asked of every agent at the end of the run, whether or not they ever opened the app, '
      + 'so this pass also carries the opinions of agents who only saw it go past in a feed.',
    ]),
    el('div', { class: 'stats' }, [
      stat(pct(interviews.would_use_rate), 'would use'),
      stat(pct(interviews.would_share_rate), 'would share'),
      stat(mean(interviews.delight_mean), 'delight'),
      stat(pct(interviews.audience_fit_rate), 'for me'),
    ]),
  ]));

  // Per agent the two passes are merged, trial first -- so the cell is marked
  // with where its number came from rather than quietly presenting a fallback
  // as if the agent had used the app.
  const src = (t, i) => (t != null ? 'trial' : i != null ? 'interview' : null);
  const cell = (t, i, render = String) => {
    const from = src(t, i);
    return el('td', {
      class: from === 'interview' ? 'faint' : '',
      title: from ? `from the ${from} verdict` : 'not recorded',
      text: from ? render(t ?? i) : '—',
    });
  };

  box.append(el('table', { class: 'grid' }, [
    el('thead', {}, [el('tr', {}, [
      ['agent', ''], ['tier', ''],
      ['use', 'would use: trial verdict, falling back to the interview'],
      ['share', 'would share: trial verdict, falling back to the interview'],
      ['♦', 'delight 0–10: trial verdict, falling back to the interview'],
      ['craft', 'craft 0–10: trial verdict only'],
      ['steps', 'how many steps the agent took in the app'],
      ['reached', 'whether the app answered when the agent opened it'],
    ].map(([h, tip]) => el('th', tip ? { text: h, title: tip } : { text: h })))]),
    el('tbody', {}, state.agents.filter((a) => a.id !== 0).map((a) => {
      const t = a.trial || {};
      const i = a.interview || {};
      const use = t.would_use ?? i.would_use;
      return el('tr', { class: 'clickable', onclick: () => selectAgent(a.id) }, [
        el('td', { text: `@${a.username}` }),
        el('td', { class: 'faint', text: a.tier || '' }),
        el('td', {
          class: use === true ? 'ok-text' : use === false ? 'err-text' : 'faint',
          title: src(t.would_use, i.would_use) ? `from the ${src(t.would_use, i.would_use)} verdict` : 'not recorded',
          text: truthy(use),
        }),
        cell(t.would_share, i.would_share, truthy),
        cell(t.delight, i.delight),
        el('td', { text: t.craft != null ? String(t.craft) : '—' }),
        el('td', { text: String(t.n_steps ?? '—') }),
        el('td', { class: t.app_reachable === false ? 'err-text' : '', text: truthy(t.app_reachable) }),
      ]);
    })),
  ]));
  return box;
}

function histogram(values, title) {
  const max = Math.max(...values, 1);
  return el('div', { style: 'padding:0 12px 12px' }, [
    el('div', { class: 'k', style: 'font-size:10px;text-transform:uppercase;color:var(--fg-faint);margin-bottom:4px', text: title }),
    el('div', { style: 'display:flex;gap:3px;align-items:flex-end;height:52px' }, values.map((count, score) =>
      el('div', {
        title: `delight ${score}: ${count} agent(s)`,
        style: `flex:1;background:${count ? 'var(--accent)' : 'var(--bg-3)'};height:${Math.max((count / max) * 100, 3)}%;border-radius:2px 2px 0 0`,
      }))),
    el('div', { style: 'display:flex;gap:3px' }, values.map((_, score) =>
      el('div', { class: 'faint center', style: 'flex:1;font-size:9px', text: String(score) }))),
  ]);
}

function viewAutorating() {
  const rating = state.run.autorating || {};
  const dims = rating.dimensions || {};
  if (!Object.keys(dims).length) return empty('No autorating recorded for this run.');
  const box = el('div');
  box.append(el('div', { class: 'section', style: 'background:transparent;border:none' }, [
    el('h2', {}, [`LLM autorater: ${rating.model || ''}, ${rating.repeats || 1} repeats`]),
    el('div', { class: 'stats' }, Object.entries(dims).map(([name, d]) =>
      stat(d.score != null ? String(d.score) : '—', name.replace(/_/g, ' ')))),
  ]));
  for (const [name, d] of Object.entries(dims)) {
    box.append(fold(el('span', {}, [el('b', { text: name.replace(/_/g, ' ') }),
      el('span', { class: 'badge', style: 'margin-left:8px', text: `score ${d.score}` }),
      d.spread != null ? el('span', { class: 'badge', style: 'margin-left:4px', text: `spread ${d.spread}` }) : null,
    ].filter(Boolean)), el('div', {}, [
      el('div', { class: 'small', style: 'margin-bottom:6px', text: d.reason || '' }),
      (d.evidence || []).length ? el('div', { class: 'chips' }, d.evidence.map((ev) => {
        const agentMatch = /agent\s+(\d+)/i.exec(ev);
        const postMatch = /POST\s+(\d+)/i.exec(ev);
        return el('span', {
          class: 'chip', style: 'cursor:pointer',
          onclick: () => {
            if (agentMatch) selectAgent(Number(agentMatch[1]));
            if (postMatch) { state.selectedPost = Number(postMatch[1]); state.tab = 'post'; renderTabs(); }
          },
        }, [ev]);
      })) : null,
      d.samples ? el('div', { class: 'faint small', style: 'margin-top:6px', text: `samples: ${d.samples.join(', ')}` }) : null,
    ].filter(Boolean))));
  }
  return box;
}

function viewHealth() {
  const r = state.run;
  const box = el('div');
  box.append(kv([
    ['healthy', truthy(r.health?.ok)],
    ['failures', (r.health?.failures || []).join('; ') || 'none'],
    ['build validity', r.validity ? `builds=${truthy(r.validity.builds)} runs=${truthy(r.validity.runs)} claims=${truthy(r.validity.does_what_it_claims)}` : '—'],
    ['validity detail', r.validity?.detail],
    ['turns', r.turn_stats ? `${r.turn_stats.turns} taken · ${r.turn_stats.skipped} skipped (${r.turn_stats.skipped_rate_limited} rate-limited) · ${r.turn_stats.budget_exhausted} out of budget` : '—'],
    ['interviews', r.interview_stats ? `${r.interview_stats.mode} · ${r.interview_stats.agents_asked} asked · ${r.interview_stats.repair_passes} repair passes` : '—'],
    ['crowd integrity', r.integrity ? `${r.integrity.actual_n_agents}/${r.integrity.requested_n_agents} agents from a pool of ${r.integrity.persona_pool_size}${r.integrity.clamped ? ' (clamped)' : ''}` : '—'],
    ['skepticism mix', r.integrity?.trier_skepticism ? Object.entries(r.integrity.trier_skepticism).map(([k, v]) => `${k} ${v}`).join(' · ') : '—'],
  ]));
  const reach = r.engagement?.reach || {};
  const cascade = r.engagement?.cascade || {};
  box.append(el('div', { class: 'section', style: 'background:transparent;border:none' }, [
    el('h2', {}, ['Reach']),
    el('div', { class: 'stats' }, [
      stat(num(reach.exposed_agents), 'exposed'), stat(num(reach.impressions), 'impressions'),
      stat(num(reach.actors_liked), 'liked'), stat(num(reach.actors_reposted), 'reposted'),
      stat(num(reach.actors_commented), 'commented'), stat(num(reach.actors_negative), 'negative'),
    ]),
    el('h2', { style: 'margin-top:10px' }, ['Cascade']),
    el('div', { class: 'stats' }, Object.entries(cascade).map(([k, v]) =>
      stat(typeof v === 'number' ? (v < 1 && v > 0 ? v.toFixed(2) : String(v)) : String(v), k.replace(/_/g, ' ')))),
  ]));
  box.append(fold('Full config', el('pre', { class: 'block', text: JSON.stringify(r.config, null, 2) })));
  return box;
}

// ---------------------------------------------------------------- right pane

function renderTabs() {
  const agent = state.byId.get(state.selectedAgent);
  const tabs = [
    ['agent', agent ? `@${agent.username}` : 'Agent'],
    ['trial', `Trial replay${state.trial ? ` (${state.trial.steps.length})` : ''}`],
    ['feedof', 'What they saw'],
    ['post', 'Post'],
    ['app', 'Run the app'],
    ['paths', 'Files on disk'],
  ];
  clear($('#tabs')).append(...tabs.map(([key, label]) =>
    el('div', { class: `tab${state.tab === key ? ' active' : ''}`, onclick: () => { state.tab = key; renderTabs(); } }, [label])));
  renderTabBody();
}

function renderTabBody() {
  const body = clear($('#tabBody'));
  if (!state.run) return;
  const views = { agent: viewAgent, trial: viewTrial, feedof: viewFeedOf, post: viewPost, app: viewApp, paths: viewPaths };
  body.append((views[state.tab] || viewAgent)());
}

function viewAgent() {
  const agent = state.byId.get(state.selectedAgent);
  if (!agent) return empty('Pick an agent from the list or the graph.');
  const box = el('div');
  const t = agent.trial || {}, i = agent.interview || {};

  box.append(el('div', { class: 'detail-head' }, [
    el('div', { style: 'display:flex;gap:9px;align-items:center;flex-wrap:wrap' }, [
      el('div', { class: 'avatar', style: `background:${TIER_COLORS[agent.tier] || '#6b7488'};width:26px;height:26px;font-size:10px`, text: initials(agent.name || agent.username) }),
      el('b', { text: `@${agent.username}` }),
      el('span', { class: 'dim', text: agent.name }),
      chip(agent.tier || 'crowd'),
      agent.archetype && chip(agent.archetype),
      agent.influence != null && chip(`influence ${agent.influence}`),
    ].filter(Boolean)),
    agent.bio && el('div', { class: 'small dim', style: 'margin-top:5px', text: agent.bio }),
  ].filter(Boolean)));

  box.append(kv([
    ['followers', `${agent.followers} (follows ${agent.followings})`],
    ['posts made', String(agent.posts)],
    ['trial verdict', agent.trial ? `would use ${truthy(t.would_use)} · would share ${truthy(t.would_share)} · delight ${t.delight ?? '—'} · craft ${t.craft ?? '—'}` : '(did not get hands on the app)'],
    ['craft facets', agent.trial ? `function ${t.functionality ?? '—'} · usability ${t.usability ?? '—'} · design ${t.design ?? '—'} · simplicity ${t.simplicity ?? '—'}` : null],
    ['work survived reload', agent.trial ? truthy(t.work_survived) : null],
    ['saw other users', agent.trial ? truthy(t.saw_other_users) : null],
    ['app reachable', agent.trial ? truthy(t.app_reachable) : null],
    ['interview', agent.interview ? `would use ${truthy(i.would_use)} · delight ${i.delight ?? '—'} · for me ${truthy(i.for_me)}` : '(no interview recorded)'],
  ]));

  if (i.why) box.append(fold('Why, in their own words (exit interview)', el('pre', { class: 'block', text: i.why }), true));

  if (agent.reasoning?.length) {
    box.append(el('div', { class: 'note', style: 'margin:12px' }, [
      el('b', {}, ['This is real reasoning. ']),
      'Unlike the founder agents, the crowd\'s own narration of what it did and why is recorded ',
      'in trajectories.json, so what follows is verbatim.',
    ]));
    agent.reasoning.forEach((text, index) => {
      box.append(fold(`Reasoning ${index + 1} of ${agent.reasoning.length}`, el('pre', { class: 'block', text }), index === 0));
    });
  }
  if (agent.persona_prompt) {
    box.append(fold('The persona it was given', el('pre', { class: 'block tall', text: agent.persona_prompt })));
  }

  const theirPosts = state.run.posts.filter((p) => p.user_id === agent.id);
  if (theirPosts.length) {
    box.append(el('h2', { style: 'padding:8px 12px 0' }, [`Everything they posted (${theirPosts.length})`]));
    for (const post of theirPosts) box.append(postCard(post));
  }
  return box;
}

function viewTrial() {
  if (!state.trial) {
    const agent = state.byId.get(state.selectedAgent);
    return empty(agent ? `@${agent.username} never got hands on the app (no trial recorded).` : 'Pick an agent.');
  }
  const trial = state.trial;
  const steps = trial.steps;
  const current = steps[state.trialIndex];
  const box = el('div');

  box.append(el('div', { class: 'detail-head' }, [
    el('div', { class: 'chips' }, [
      el('button', { class: 'primary', onclick: () => trialPlayback.toggle() }, [trialPlayback.playing ? '❚❚ Pause' : '▶ Replay the trial']),
      el('button', { onclick: () => { trialPlayback.pause(); jumpStep(state.trialIndex - 1); } }, ['◀']),
      el('button', { onclick: () => { trialPlayback.pause(); jumpStep(state.trialIndex + 1); } }, ['▶']),
      chip(`step ${state.trialIndex + 1} / ${steps.length}`),
      chip(trial.app_type || ''),
      trial.degraded && chip('degraded (no browser)', 'warn'),
      trial.app_reachable === false && chip('app unreachable', 'err'),
      trial.target_url && el('a', { href: trial.target_url, target: '_blank', class: 'chip', text: trial.target_url }),
    ].filter(Boolean)),
    el('div', { class: 'small dim', style: 'margin-top:5px' }, [
      `${dur((trial.ended_at || 0) - (trial.started_at || 0))} of real time · this is exactly what the agent did to the app`,
    ]),
  ]));

  const list = el('div', { style: 'max-height:210px;overflow:auto;border-bottom:1px solid var(--line)' });
  let selectedRow = null;
  steps.forEach((step, index) => {
    const row = el('div', {
      class: `step${index === state.trialIndex ? ' sel' : ''}${step.ok === false ? ' fail' : ''}`,
      onclick: () => { trialPlayback.pause(); jumpStep(index); },
    }, [
      el('div', { class: 'faint', text: String(index) }),
      el('div', { class: `verb v-${step.action}`, text: step.action }),
      el('div', { class: 'what' }, [describeStep(step)]),
    ]);
    if (index === state.trialIndex) selectedRow = row;
    list.append(row);
  });
  box.append(list);
  // The window is fixed-height over a trial that can run 40 steps, so during
  // playback the current step has to scroll itself into view.
  if (selectedRow) requestAnimationFrame(() => selectedRow.scrollIntoView({ block: 'nearest' }));

  if (!current) return box;

  if (current.screenshot) {
    box.append(el('div', { style: 'padding:10px 12px 0' }, [
      el('img', {
        class: 'shot', src: `/api/shot/${encodeURIComponent(current.screenshot)}`,
        onerror: (e) => e.target.replaceWith(el('div', { class: 'faint small', text: `screenshot no longer on disk: ${current.screenshot_path}` })),
      }),
      el('div', { class: 'faint small', style: 'margin-top:3px', text: `${current.args?.note || ''} · ${current.screenshot}` }),
    ]));
  } else {
    // Screenshots only exist on explicit screenshot steps, so for every other step
    // show the most recent one as the visual context the agent was acting on.
    const previous = steps.slice(0, state.trialIndex).reverse().find((s) => s.screenshot);
    if (previous) {
      box.append(el('div', { style: 'padding:10px 12px 0' }, [
        el('img', { class: 'shot', style: 'opacity:.55', src: `/api/shot/${encodeURIComponent(previous.screenshot)}`, onerror: (e) => e.target.remove() }),
        el('div', { class: 'faint small', style: 'margin-top:3px', text: `last screenshot taken (step ${previous.index}), the agent had no camera on this step` }),
      ]));
    }
  }

  box.append(kv([
    ['action', current.action],
    ['arguments', JSON.stringify(current.args)],
    ['ok', truthy(current.ok)],
    ['took', current.duration_s != null ? `${current.duration_s.toFixed(2)}s` : '—'],
    ['errors', (current.errors || []).join('; ') || null],
  ]));
  box.append(fold(`What the app showed back${current.summary_chars > current.summary.length ? ` (first ${bytes(current.summary.length)} of ${bytes(current.summary_chars)})` : ''}`,
    el('pre', { class: 'block tall', text: current.summary }), true));

  if (trial.verdict) {
    box.append(fold('The verdict it left', el('pre', { class: 'block', text: JSON.stringify(trial.verdict, null, 2) })));
  }
  return box;
}

function describeStep(step) {
  const a = step.args || {};
  if (step.action === 'open' || step.action === 'reload') return a.url || '';
  if (step.action === 'click') return a.target || '';
  if (step.action === 'type') return `${a.target || ''} ← "${String(a.text || '').slice(0, 60)}"`;
  if (step.action === 'press') return `${a.key}${a.target ? ` on ${a.target}` : ''}`;
  if (step.action === 'select') return `${a.target} = ${a.value}`;
  if (step.action === 'screenshot') return a.note || '';
  if (step.action === 'run_command') return a.command || '';
  if (step.action === 'send_message') return String(a.message || '').slice(0, 70);
  if (step.action === 'finish') return `would_use=${a.would_use} delight=${a.delight} craft=${a.craft ?? '—'}`;
  return '';
}

function jumpStep(index) {
  const steps = state.trial?.steps || [];
  if (!steps.length) return;
  state.trialIndex = Math.max(0, Math.min(steps.length - 1, index));
  const ts = steps[state.trialIndex].ts;
  if (ts) trialPlayback.value = ts;
  renderTabBody();
}

function viewFeedOf() {
  const agent = state.byId.get(state.selectedAgent);
  if (!agent) return empty('Pick an agent.');
  const feeds = state.run.feeds[String(agent.id)] || {};
  const steps = Object.keys(feeds).map(Number).sort((a, b) => a - b);
  const box = el('div');
  box.append(el('div', { class: 'note', style: 'margin:12px' }, [
    el('b', {}, ['This is exposure, not the whole timeline. ']),
    'The recommender decides what each agent is shown, and this is the feed ',
    el('b', {}, [`@${agent.username}`]), ' saw at each step, the only time-resolved record of who was ',
    'given a chance to see the launch at all.',
  ]));
  if (!steps.length) return box.append(empty('This agent never refreshed a feed.')), box;

  for (const step of steps) {
    const ids = feeds[step];
    box.append(fold(`t${step} · ${ids.length} post(s) in feed${ids.includes(state.run.launch_post_id) ? ' · included the launch post' : ''}`,
      el('div', { class: 'faint small' }, ['loading…']), step === steps[steps.length - 1]));
    const target = box.lastChild;
    target.addEventListener('toggle', async function once() {
      if (!target.open || target.dataset.loaded) return;
      target.dataset.loaded = '1';
      try {
        const data = await api(`/api/crowd/${encodeURIComponent(state.run.run_id)}/feed/${agent.id}/${step}`);
        const body = clear(target.querySelector('.fold-body'));
        if (!data.posts.length) { body.append(empty('empty feed')); return; }
        for (const p of data.posts) {
          body.append(el('div', { class: 'post', style: 'margin:6px 0' }, [
            el('div', { class: 'post-head' }, [
              el('span', { class: 'who', text: p.author || `user ${p.user_id}` }),
              el('span', { class: 'faint', text: `#${p.post_id} · t${p.created_at}` }),
            ]),
            el('div', { class: 'post-body', text: String(p.content || '').slice(0, 500) }),
            el('div', { class: 'post-meta' }, [`♥ ${p.num_likes}`, `↻ ${p.num_shares}`, `💬 ${(p.comments || []).length}`]),
          ]));
        }
      } catch (err) {
        clear(target.querySelector('.fold-body')).append(el('div', { class: 'err-text small', text: String(err.message || err) }));
      }
    });
    if (target.open) target.dispatchEvent(new Event('toggle'));
  }
  return box;
}

function viewPost() {
  const post = state.run.posts.find((p) => p.post_id === state.selectedPost);
  if (!post) return empty('Click a post in the feed to see who saw it and who acted on it.');
  const author = state.byId.get(post.user_id);
  const box = el('div');
  box.append(el('div', { class: 'detail-head' }, [
    el('b', { text: `Post #${post.post_id}` }),
    el('span', { class: 'dim', text: ` by @${author?.username || post.user_id} at t${post.t}` }),
  ]));
  box.append(postCard(post));

  const shown = Object.entries(state.run.rec).filter(([, ids]) => ids.includes(post.post_id)).map(([id]) => Number(id));
  const likers = state.run.likes.filter((l) => l.post_id === post.post_id).map((l) => l.user_id);
  const amplifiers = state.run.actions.filter((a) => a.source_post === post.post_id);

  box.append(kv([
    ['shown to', `${shown.length} agent(s) in the final recommender snapshot`],
    ['liked by', `${likers.length}`],
    ['amplified by', `${amplifiers.length} (${amplifiers.filter((a) => a.action === 'repost').length} reposts, ${amplifiers.filter((a) => a.action === 'quote_post').length} quotes)`],
  ]));

  const roster = (ids, title) => ids.length ? fold(`${title} (${ids.length})`, el('div', { class: 'chips' },
    [...new Set(ids)].map((id) => el('span', {
      class: 'chip', style: 'cursor:pointer', onclick: () => selectAgent(id),
    }, [`@${state.byId.get(id)?.username || id}`])))) : null;

  box.append(roster(likers, 'Liked by'));
  box.append(roster(amplifiers.map((a) => a.user_id), 'Amplified by'));
  box.append(roster(shown, 'Was in the feed of'));
  return box;
}

function viewApp() {
  const r = state.run;
  // The tab body scrolls, so height:100% has nothing to resolve against, and a
  // min-height keeps the app pane usable at any window size.
  const box = el('div', { style: 'display:flex;flex-direction:column;min-height:100%' });
  const bar = el('div', { class: 'section', style: 'background:transparent' });
  const frame = el('div', { style: 'flex:1 1 auto;min-height:70vh;background:#fff' });
  const status = el('div', { class: 'small dim', style: 'margin-top:6px' });
  box.append(bar);

  const launch = el('button', {
    class: 'primary',
    onclick: async () => {
      launch.disabled = true;
      status.textContent = 'materialising a throwaway copy, running setup, starting the app…';
      try {
        const res = await post('/api/app/launch', { build_id: r.build_id });
        if (!res.ok) {
          status.replaceChildren(el('div', { class: 'err-text', text: res.error }),
            res.log ? el('pre', { class: 'block', text: res.log }) : el('span'));
          launch.disabled = false;
          return;
        }
        state.app = res;
        status.replaceChildren(el('span', {}, ['running at ', el('a', { href: res.url, target: '_blank', text: res.url })]));
        clear(frame).append(el('iframe', { class: 'appframe', src: res.url }));
        launch.textContent = 'Restart';
        launch.disabled = false;
        stopBtn.disabled = false;
      } catch (err) {
        status.replaceChildren(el('span', { class: 'err-text', text: String(err.message || err) }));
        launch.disabled = false;
      }
    },
  }, ['▶ Launch the app the crowd tested']);

  const stopBtn = el('button', {
    disabled: true,
    onclick: async () => {
      if (!state.app) return;
      await post('/api/app/stop', { session_id: state.app.session_id }).catch(() => {});
      state.app = null; clear(frame); status.textContent = 'stopped';
      stopBtn.disabled = true; launch.textContent = '▶ Launch the app the crowd tested';
    },
  }, ['■ Stop']);

  bar.append(
    el('h2', {}, ['Use the app yourself']),
    el('div', { class: 'chips' }, [launch, stopBtn, chip(r.app_type || '?'), chip(r.build_id)]),
    el('div', { class: 'small dim', style: 'margin-top:6px' }, [
      'Same build the agents were given. Open the Trial replay tab beside this and you can repeat, ',
      'by hand, the exact steps an agent took, and judge its verdict for yourself.',
    ]),
    status,
  );
  box.append(frame);
  return box;
}

function viewPaths() {
  const r = state.run;
  return el('div', {}, [
    el('div', { class: 'note', style: 'margin:12px' }, [
      el('b', {}, ['Everything is already on disk. ']),
      'The run directory below holds the raw JSON. Download bundles it into one file; the zip adds ',
      'the per-agent trials and whatever screenshots still resolve.',
    ]),
    kv([
      ['run directory', el('span', { class: 'mono', text: r.paths.run_dir })],
      ['summary', el('span', { class: 'mono', text: r.paths.summary })],
      ['action log', el('span', { class: 'mono', text: r.paths.actions })],
      ['sqlite db', el('span', { class: 'mono', text: r.paths.db })],
      ['per-agent trials', el('span', { class: 'mono', text: r.paths.traces })],
      ['founder build', el('span', { class: 'mono', text: r.build_id })],
    ]),
    el('div', { class: 'note', style: 'margin:12px' }, [
      el('b', {}, ['Screenshots are the exception. ']),
      'The crowd browser wrote them to /tmp/viralbench-shots, which does not survive a reboot. ',
      '"Save shots" in the toolbar copies this run\'s images into the viewer cache so they keep working.',
    ]),
  ]);
}

// ---------------------------------------------------------------- browser

async function openBrowser(buildId) {
  $('#browser').hidden = false;
  if (buildId) $('#searchInput').value = buildId;
  $('#searchInput').focus();
  await refreshBrowse();
}

async function refreshBrowse() {
  const query = new URLSearchParams({ limit: '400' });
  const search = $('#searchInput').value.trim();
  if (search) query.set('q', search);
  if ($('#okOnly').checked) query.set('ok', '1');
  const body = $('#browseBody');
  body.replaceChildren(el('div', { class: 'empty' }, [el('span', { class: 'spin', text: '◐' }), ' searching…']));
  try {
    const result = await api(`/api/crowd/runs?${query}`);
    body.replaceChildren(
      el('div', { class: 'small dim', style: 'padding:6px 14px', text: `${result.matched} runs matched (showing ${result.runs.length})` }),
      el('table', { class: 'grid' }, [
        el('thead', {}, [el('tr', {}, ['build', 'app type', 'agents', 'rounds', 'posts', '♥', '💬', '↻', 'use', '♦', 'arch', 'trials', 'run id']
          .map((h) => el('th', { text: h })))]),
        el('tbody', {}, result.runs.map((row) => el('tr', {
          class: 'clickable',
          onclick: () => { $('#browser').hidden = true; load(row.run_id); },
        }, [
          el('td', { text: row.build_id.split('__')[0] }),
          el('td', { class: row.app_type.includes('app') ? 'ok-text' : 'faint', text: row.app_type }),
          el('td', { text: String(row.n_agents ?? '—') }),
          el('td', { text: String(row.rounds ?? '—') }),
          el('td', { text: String(row.posts) }),
          el('td', { text: String(row.likes) }),
          el('td', { text: String(row.comments) }),
          el('td', { text: String(row.reposts) }),
          el('td', { text: row.would_use_rate != null ? `${Math.round(row.would_use_rate * 100)}%` : '—' }),
          el('td', { text: row.delight_mean != null ? row.delight_mean.toFixed(1) : '—' }),
          el('td', { class: 'faint', text: `v${row.arch_version}` }),
          el('td', { text: String(row.n_traces) }),
          el('td', { class: 'faint', text: row.run_id }),
        ]))),
      ]));
  } catch (err) {
    body.replaceChildren(el('div', { class: 'empty err-text', text: String(err.message || err) }));
  }
}

// ---------------------------------------------------------------- wiring

$('#loadBtn').addEventListener('click', () => load($('#runInput').value.trim()));
$('#runInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') load($('#runInput').value.trim()); });
$('#browseBtn').addEventListener('click', () => openBrowser());
$('#closeBrowse').addEventListener('click', () => { $('#browser').hidden = true; });
$('#browser').addEventListener('click', (e) => { if (e.target.id === 'browser') $('#browser').hidden = true; });
$('#founderBtn').addEventListener('click', () => { location.href = `/founder?build=${encodeURIComponent(state.run.build_id)}`; });
$('#dlJson').addEventListener('click', () => { location.href = `/api/crowd/${encodeURIComponent(state.run.run_id)}/download`; });
$('#dlZip').addEventListener('click', () => { location.href = `/api/crowd/${encodeURIComponent(state.run.run_id)}/download?format=zip`; });
$('#rescueBtn').addEventListener('click', async () => {
  $('#rescueBtn').disabled = true;
  try {
    const res = await post(`/api/crowd/${encodeURIComponent(state.run.run_id)}/rescue`);
    toast(`${res.rescued} of ${res.referenced} screenshots saved into the viewer cache${res.missing ? ` · ${res.missing} already gone` : ''}`);
  } catch (err) { toast(String(err.message || err), true); }
  $('#rescueBtn').disabled = false;
});

$('#graph').addEventListener('click', (e) => { const id = graphHitTest(e); if (id !== null) selectAgent(id); });
$('#graph').addEventListener('mousemove', (e) => {
  const id = graphHitTest(e);
  $('#graph').style.cursor = id !== null ? 'pointer' : 'default';
  if (id !== state.hoverAgent) { state.hoverAgent = id; if (state.run) drawGraph(); }
});
$('#graph').addEventListener('mouseleave', () => { state.hoverAgent = null; if (state.run) drawGraph(); });
window.addEventListener('resize', () => { if (state.run) drawGraph(); });

let searchTimer = null;
$('#searchInput').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(refreshBrowse, 220); });
$('#okOnly').addEventListener('change', refreshBrowse);

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') { if (e.key === 'Escape') $('#browser').hidden = true; return; }
  if (e.key === ' ') { e.preventDefault(); playback.toggle(); }
  else if (e.key === 'ArrowRight') { playback.pause(); playback.seek(Math.floor(playback.value) + 1); }
  else if (e.key === 'ArrowLeft') { playback.pause(); playback.seek(Math.floor(playback.value) - 1); }
  else if (e.key === 'Escape') $('#browser').hidden = true;
  else if (e.key === '/') { e.preventDefault(); openBrowser(); }
});

clear($('#legend')).append(...Object.entries(TIER_COLORS).map(([tier, color]) =>
  el('span', {}, [el('i', { style: `background:${color}` }), tier])),
  el('span', { class: 'faint' }, ['· node size = engagement received · green/pink arcs = a repost or quote happening now']));

const initialRun = params.get('run');
const initialBuild = params.get('build');
if (initialRun) load(initialRun);
else if (initialBuild) openBrowser(initialBuild);
else openBrowser();
