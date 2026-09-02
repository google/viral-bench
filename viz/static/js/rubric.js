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

/* Rubric grade viewer.
 *
 * The other two viewers replay a recording. This one renders a *verdict
 * document*, and the difference drives the whole design: there is no clock, no
 * scrubber, and nothing to play. What matters instead is that every number can
 * be traced back to the thing that produced it.
 *
 * So two rules run through this file.
 *
 * The page never recomputes a score. `grade.math` carries every figure the
 * header shows, already worked out by the scorer. Re-deriving `earned/applicable`
 * here would be a second implementation of the arithmetic, and two
 * implementations of one number is one too many -- they disagree eventually, and
 * the UI is the one people believe.
 *
 * The page distinguishes "failed" from "could not tell". An unresolved item and
 * a failed item both earn zero, but they mean opposite things about the app: one
 * is a defect, the other is an instrument fault. They are rendered differently
 * everywhere, because collapsing them is how a flaky grader starts looking like
 * a bad app.
 */

import {
  $, api, chip, clear, el, empty, fold, kv, params, stat, toast,
} from './common.js';

const state = {
  grade: null,
  runId: '',
  rows: [],
  selected: null,
  tab: 'item',
  filter: 'all',
  transcript: null,
};

const ARM_CLASS = { solo: 'accent', team: 'ok', dynamic: 'warn' };

// A grade is decided either by code or by the model, and which one it was is the
// single most load-bearing fact about a verdict's trustworthiness.
const METHOD_CLASS = { assert: 'ok', probe: 'ok', source: 'accent', agent: 'warn' };

// ---------------------------------------------------------------- loading

async function load(id) {
  if (!id) return;
  $('#items').replaceChildren(el('div', { class: 'empty' }, [
    el('span', { class: 'spin', text: '◐' }), ' loading…',
  ]));
  // A build id and a grade id are both plausible pastes. If it is a build, list
  // its grades rather than erroring -- same tolerance the crowd viewer has.
  try {
    adopt(await api(`/api/rubric/${encodeURIComponent(id)}`));
  } catch {
    try {
      const listed = await api(
        `/api/rubric/grades?build_id=${encodeURIComponent(id)}&limit=200`);
      if (listed.grades.length === 1) return load(listed.grades[0].run_id);
      if (listed.grades.length) { openBrowser(id); return; }
      toast(`No rubric grade for "${id}" yet`, true);
    } catch (err) { toast(String(err.message || err), true); }
    $('#items').replaceChildren(empty('Nothing loaded.'));
  }
}

function adopt(grade) {
  state.grade = grade;
  state.runId = grade.run_id;
  state.rows = flatten(grade);
  state.selected = state.rows.find((r) => !r.passed) || state.rows[0] || null;
  state.tab = 'item';
  state.transcript = null;
  $('#gradeInput').value = grade.run_id;
  params.set('grade', grade.run_id);

  const buildId = grade.build_id || '';
  for (const [sel, href] of [
    ['#founderBtn', `/founder?build=${encodeURIComponent(buildId)}`],
    ['#crowdBtn', `/crowd?build=${encodeURIComponent(buildId)}`],
  ]) {
    const button = $(sel);
    button.disabled = !buildId;
    button.onclick = () => { location.href = href; };
  }
  $('#dlJson').disabled = false;
  $('#dlZip').disabled = false;

  renderHead();
  renderFilters();
  renderItems();
  renderTabs();
}

/** Every gate item, scored item and penalty, in one list, in rubric order. */
function flatten(grade) {
  const rows = [];
  for (const item of grade.gate?.items || []) {
    rows.push({ ...item, tier: 0, kind: 'gate', points: 0, earned: 0 });
  }
  for (const tier of grade.tiers || []) {
    for (const item of tier.items || []) {
      rows.push({ ...item, tier: tier.tier, kind: 'item' });
    }
  }
  for (const penalty of grade.penalties || []) {
    rows.push({
      ...penalty,
      tier: -1,
      kind: 'penalty',
      passed: penalty.fired,
      earned: penalty.fired ? penalty.points : 0,
    });
  }
  return rows;
}

// ---------------------------------------------------------------- header

function renderHead() {
  const g = state.grade;
  const math = g.math || {};
  const f = g.founder || {};
  const cmp = g.comparison;
  const rel = g.reliability || {};
  const gateFailed = !!math.gate_zeroed;

  const chips = [
    el('span', {
      class: `chip mode ${ARM_CLASS[f.arm] || ''}`,
      style: f.build_id ? 'cursor:pointer' : '',
      title: `Open this build in the founder viewer\n${f.build_id || ''}`,
      onclick: () => f.build_id
        && (location.href = `/founder?build=${encodeURIComponent(f.build_id)}`),
    }, [`built by: ${f.arm_label || 'unknown'}`]),
    chip(f.model_short || 'unknown model', 'accent'),
    chip(g.idea_id || 'unknown idea'),
    el('span', { class: 'faint', style: 'margin:0 2px' }, ['|']),
    chip(gateFailed ? 'gate: FAILED' : 'gate: passed', gateFailed ? 'err' : 'ok'),
    chip(`grader: ${(g.grader_model || '').split('@')[0] || '?'}`, 'purple'),
    chip(`${g.passes ?? '?'} passes`),
    chip(`rubric v${g.rubric_version || '?'}`),
    // Not decoration: if the harness kept overruling the model, the transcript
    // is not trustworthy even where the code had the last word.
    rel.override_rate > 0.1
      && chip(`override rate ${pct(rel.override_rate)}`, 'err'),
    rel.items_disagreeing > 0
      && chip(
        `${rel.items_disagreeing} item${rel.items_disagreeing === 1 ? '' : 's'} `
        + 'disagreed across passes', 'warn'),
    rel.unresolved > 0 && chip(`${rel.unresolved} unresolved`, 'warn'),
  ].filter(Boolean);

  const stats = [
    stat(fmt(g.score), 'RubricScore', {
      title: 'Out of 100. base + penalties, floored at 0. See the Score math tab.',
      onclick: () => { state.tab = 'math'; renderTabs(); },
    }),
    stat(`${math.points_earned ?? '?'}/${math.points_applicable ?? '?'}`, 'points', {
      title: 'Earned over APPLICABLE points. Not-applicable items leave the '
        + 'denominator, so ideas stay comparable.',
      onclick: () => { state.tab = 'math'; renderTabs(); },
    }),
    stat(fmt(math.base), 'base', { title: '100 x earned / applicable' }),
    stat(
      `${(math.penalty_total ?? 0) > 0 ? '+' : ''}${math.penalty_total ?? 0}`,
      math.penalty_capped ? 'penalties (capped)' : 'penalties',
      { title: 'Capped at -25 in total.' },
    ),
  ];
  if (cmp) {
    const delta = g.score != null && cmp.viral_score_mean != null
      ? Math.round((g.score - cmp.viral_score_mean) * 10) / 10
      : null;
    stats.push(stat(fmt(cmp.viral_score_mean), 'ViralScore', {
      title: `Mean of ${cmp.crowd_runs} crowd run(s): `
        + `${fmt(cmp.viral_score_min)}–${fmt(cmp.viral_score_max)}`,
      onclick: () => { state.tab = 'versus'; renderTabs(); },
    }));
    if (delta !== null) {
      stats.push(stat(`${delta > 0 ? '+' : ''}${delta}`, 'rubric − crowd', {
        title: 'Positive: the rubric rates it higher than the crowd did.',
        onclick: () => { state.tab = 'versus'; renderTabs(); },
      }));
    }
  }

  clear($('#head')).append(
    el('div', { style: 'display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap' }, [
      el('div', { style: 'flex:1 1 340px;min-width:0' }, [
        el('div', { style: 'display:flex;gap:9px;align-items:baseline;flex-wrap:wrap' }, [
          el('div', { style: 'font-size:16px;font-weight:700' }, [
            f.app_title || (g.build_id || '').split('__')[0],
          ]),
          el('div', { class: 'dim small' }, [(g.build_id || '').split('__')[0]]),
          el('div', { class: 'mono faint small', text: g.run_id }),
        ]),
        el('div', { class: 'chips', style: 'margin-top:6px' }, chips),
      ]),
      el('div', { class: 'stats' }, stats),
    ]),
  );

  // When the gate failed, every other number on the page is zero. Say why, or
  // the page reads as "this app scored nothing on 27 separate counts".
  if (gateFailed) {
    const failed = (g.gate?.items || []).filter((i) => i.passed === false);
    $('#head').append(el('div', { class: 'note', style: 'margin-top:8px' }, [
      el('strong', {}, ['Deliverability gate failed, scored 0. ']),
      failed.length
        ? `${failed[0].id}: ${failed[0].detail || failed[0].text}`
        : 'The build could not be run.',
      el('div', { class: 'small dim', style: 'margin-top:4px' }, [
        'The graded items below were never observed, so they are shown '
        + 'unresolved rather than failed.',
      ]),
    ]));
  }
}

// ---------------------------------------------------------------- item list

const FILTERS = [
  ['all', 'All'],
  ['failed', 'Failed'],
  ['passed', 'Passed'],
  ['unresolved', 'Unresolved'],
  ['disagreed', 'Disagreed'],
  ['penalty', 'Penalties'],
];

function renderFilters() {
  clear($('#itemFilters')).append(...FILTERS.map(([key, label]) => el('span', {
    class: `chip${state.filter === key ? ' accent' : ''}`,
    style: 'cursor:pointer',
    onclick: () => { state.filter = key; renderFilters(); renderItems(); },
  }, [label])));
}

function visibleRows() {
  return state.rows.filter((row) => {
    switch (state.filter) {
      case 'failed': return row.passed === false;
      case 'passed': return row.passed === true && row.kind !== 'penalty';
      case 'unresolved': return row.unresolved === true || row.passed === null;
      case 'disagreed': return row.disagreement === true;
      case 'penalty': return row.kind === 'penalty';
      default: return true;
    }
  });
}

function verdictCell(row) {
  if (row.kind === 'penalty') {
    return row.passed
      ? el('span', { class: 'err-text', text: 'FIRED' })
      : el('span', { class: 'faint', text: '—' });
  }
  if (row.unresolved || row.passed === null) {
    // Third state on purpose: "could not tell" is not "wrong".
    return el('span', { class: 'faint', text: 'UNRESOLVED', title: row.reason || '' });
  }
  return row.passed
    ? el('span', { class: 'ok-text', text: 'PASS' })
    : el('span', { class: 'err-text', text: 'FAIL' });
}

function renderItems() {
  const rows = visibleRows();
  $('#itemCount').textContent = `${rows.length} of ${state.rows.length}`;
  const body = clear($('#items'));
  if (!rows.length) { body.append(empty('No items match this filter.')); return; }

  const na = new Map((state.grade.not_applicable || []).map((n) => [n.id, n.reason]));
  const table = el('table', { class: 'grid' }, [
    el('thead', {}, [el('tr', {}, [
      el('th', {}, ['id']), el('th', {}, ['item']), el('th', {}, ['pts']),
      el('th', {}, ['method']), el('th', {}, ['verdict']),
    ])]),
  ]);
  const tbody = el('tbody');
  let lastTier = null;
  for (const row of rows) {
    if (row.tier !== lastTier) {
      lastTier = row.tier;
      tbody.append(el('tr', {}, [
        el('td', { colspan: 5, style: 'background:var(--bg-2);font-weight:600' },
          [tierLabel(row.tier)]),
      ]));
    }
    const selected = state.selected && state.selected.id === row.id;
    tbody.append(el('tr', {
      style: `cursor:pointer${selected ? ';background:var(--bg-3)' : ''}`,
      onclick: () => { state.selected = row; state.tab = 'item'; renderItems(); renderTabs(); },
    }, [
      el('td', { class: 'mono' }, [row.id]),
      el('td', {}, [
        el('div', {}, [truncate(row.text, 90)]),
        row.harness_override && el('span', { class: 'chip err', style: 'margin-top:3px' },
          ['harness overrode the model']),
        row.disagreement && el('span', { class: 'chip warn', style: 'margin-top:3px' },
          ['passes disagreed']),
      ].filter(Boolean)),
      el('td', { class: 'mono nowrap' }, [
        row.kind === 'penalty'
          ? String(row.points)
          : `${row.earned ?? 0}/${row.points ?? 0}`,
      ]),
      el('td', {}, [chip(row.method || '?', METHOD_CLASS[row.method] || '')]),
      el('td', {}, [verdictCell(row)]),
    ]));
  }
  table.append(tbody);
  body.append(table);

  if (na.size) {
    body.append(el('div', { class: 'section', style: 'border-top:1px solid var(--line)' }, [
      el('h2', {}, ['Not applicable']),
      el('div', { class: 'small dim' }, [
        'Excluded from the denominator, so this idea stays comparable with the others.',
      ]),
      ...[...na].map(([id, reason]) => el('div', { class: 'small', style: 'margin-top:4px' }, [
        el('span', { class: 'mono faint' }, [`${id} `]), reason,
      ])),
    ]));
  }
}

function tierLabel(tier) {
  if (tier === 0) return 'Tier 0 · Deliverability gate';
  if (tier === -1) return 'Penalties';
  const found = (state.grade.tiers || []).find((t) => t.tier === tier);
  return found
    ? `Tier ${tier} · ${found.label}  (${found.earned}/${found.points})`
    : `Tier ${tier}`;
}

// ---------------------------------------------------------------- tabs

function renderTabs() {
  const g = state.grade;
  const tabs = [
    ['item', state.selected ? `Item ${state.selected.id}` : 'Item'],
    ['math', 'Score math'],
    ['versus', g.comparison ? 'vs ViralScore' : 'vs ViralScore (none)'],
    ['transcript', 'Transcript'],
    ['rubric', 'Rubric'],
    ['paths', 'Files on disk'],
  ];
  clear($('#tabs')).append(...tabs.map(([key, label]) => el('div', {
    class: `tab${state.tab === key ? ' active' : ''}`,
    onclick: () => { state.tab = key; renderTabs(); },
  }, [label])));
  renderTabBody();
}

function renderTabBody() {
  const body = clear($('#tabBody'));
  if (!state.grade) return;
  const views = {
    item: viewItem, math: viewMath, versus: viewVersus,
    transcript: viewTranscript, rubric: viewRubric, paths: viewPaths,
  };
  body.append((views[state.tab] || viewItem)());
}

function viewItem() {
  const row = state.selected;
  if (!row) return empty('Select an item on the left.');
  const evidence = (state.grade.evidence_index || {})[row.id] || row.evidence || [];

  const passes = (row.passes || []).map((value, index) => el('span', {
    class: `chip ${value === true ? 'ok' : value === false ? 'err' : ''}`,
    title: `pass ${index + 1}`,
  }, [value === true ? 'pass' : value === false ? 'fail' : 'unknown']));

  return el('div', { class: 'detail' }, [
    el('div', { class: 'detail-head' }, [
      el('span', { class: 'mono' }, [row.id]), ' ',
      chip(row.method || '?', METHOD_CLASS[row.method] || ''),
      row.kind === 'penalty'
        ? chip(`${row.points} pts if fired`, 'err')
        : chip(`${row.earned ?? 0} of ${row.points ?? 0} pts`,
          row.passed ? 'ok' : ''),
    ]),
    el('p', {}, [row.text]),

    row.harness_override && el('div', { class: 'note' }, [
      el('strong', {}, ['The harness overruled the model here. ']),
      'A code check decided this item and disagreed with the verdict the grader '
      + 'stated. The code result is what was scored.',
    ]),

    kv([
      ['verdict', verdictCell(row)],
      ['expected', row.expect || row.expected || ''],
      ['observed', row.observed ? el('code', { class: 'mono' }, [row.observed]) : ''],
      ['reason', row.reason || ''],
      ['per-pass', passes.length ? el('span', { class: 'chips' }, passes) : ''],
      ['agreement', row.disagreement
        ? el('span', { class: 'warn-text' }, ['passes disagreed, majority taken'])
        : (row.passes || []).length > 1 ? 'unanimous' : ''],
      ['note', row.note || ''],
    ]),

    el('h3', {}, [`Evidence (${evidence.length})`]),
    evidence.length
      ? el('div', { class: 'chips' }, evidence.map((id) => el('span', {
        class: 'chip accent mono', style: 'cursor:pointer',
        title: 'Show this call in the transcript',
        onclick: () => { state.tab = 'transcript'; state.transcript = null; renderTabs(); },
      }, [id])))
      // Not a cosmetic absence: an agent-judged verdict with no recorded call is
      // recorded FAIL by the harness, so the emptiness is the finding.
      : el('div', { class: 'small dim' }, [
        row.method === 'agent'
          ? 'No recorded tool call backs this verdict.'
          : 'Decided by code; no model evidence needed.',
      ]),
  ].filter(Boolean));
}

function bar(earned, total) {
  const pctv = total ? Math.max(0, Math.min(100, (earned / total) * 100)) : 0;
  return el('div', { class: 'meter' }, [el('div', {
    style: `width:${pctv}%;background:var(--${pctv >= 60 ? 'ok' : 'warn'})`,
  })]);
}

function viewMath() {
  const g = state.grade;
  const m = g.math || {};
  const fired = (g.penalties || []).filter((p) => p.fired);
  return el('div', { class: 'detail' }, [
    el('h3', {}, ['How this score was reached']),
    ...(g.tiers || []).map((tier) => el('div', { style: 'margin-bottom:10px' }, [
      el('div', { style: 'display:flex;justify-content:space-between' }, [
        el('span', {}, [`Tier ${tier.tier} · ${tier.label}`]),
        el('span', { class: 'mono' }, [`${tier.earned}/${tier.points}`]),
      ]),
      bar(tier.earned, tier.points),
    ])),
    (g.not_applicable || []).length ? el('div', { style: 'margin:10px 0' }, [
      el('h3', {}, ['Not applicable']),
      ...g.not_applicable.map((n) => el('div', { class: 'small' }, [
        el('span', { class: 'mono faint' }, [`${n.id} `]), n.reason,
      ])),
      el('div', { class: 'small dim', style: 'margin-top:4px' }, [
        'These leave the denominator rather than counting as failures, so an '
        + 'idea whose brief never asks for a thing is not punished for lacking it.',
      ]),
    ]) : null,
    el('h3', {}, ['Penalties']),
    fired.length
      ? el('div', {}, fired.map((p) => el('div', { class: 'small' }, [
        el('span', { class: 'mono err-text' }, [`${p.points} `]),
        el('span', { class: 'mono faint' }, [`${p.id} `]), p.text,
      ])))
      : el('div', { class: 'small dim' }, ['None fired.']),
    m.penalty_capped
      ? el('div', { class: 'note' }, ['Penalties hit the −25 cap.'])
      : null,
    el('h3', {}, ['Arithmetic']),
    el('pre', { class: 'block mono' }, [
      `base       = 100 x ${m.points_earned} / ${m.points_applicable} = ${m.base}\n`
      + `penalties  = ${m.penalty_total}${m.penalty_capped ? '  (capped at -25)' : ''}\n`
      + `score      = max(0, ${m.base} + ${m.penalty_total}) = ${g.score}`
      + (m.gate_zeroed ? '\n\nGATE FAILED -> score forced to 0' : ''),
    ]),
    el('h3', {}, ['How much to trust it']),
    kv(Object.entries(g.reliability || {}).map(([k, v]) => [
      k.replace(/_/g, ' '), typeof v === 'object' ? JSON.stringify(v) : String(v),
    ])),
  ].filter(Boolean));
}

function viewVersus() {
  const g = state.grade;
  const cmp = g.comparison;
  if (!cmp) {
    return el('div', { class: 'detail' }, [
      empty('This build has no crowd runs, so there is no ViralScore to compare.'),
      el('div', { class: 'small dim', style: 'margin-top:8px' }, [
        'The grade is still valid, but it cannot take part in the comparison.',
      ]),
    ]);
  }
  const delta = Math.round((g.score - cmp.viral_score_mean) * 10) / 10;
  const harsher = delta < 0;
  // A penalty that did not fire cost nothing, so it does not belong in a list of
  // what the rubric took points away for. Testing `passed === false` alone would
  // include it, because an unfired penalty is recorded as not-passed.
  const failed = state.rows.filter((row) => {
    if (row.kind === 'gate') return false;
    if (row.kind === 'penalty') return row.passed === true;
    return row.passed === false;
  });
  return el('div', { class: 'detail' }, [
    el('div', { class: 'stats' }, [
      stat(fmt(g.score), 'RubricScore'),
      stat(fmt(cmp.viral_score_mean), 'ViralScore', {
        title: `mean of ${cmp.crowd_runs} run(s)`,
      }),
      stat(`${delta > 0 ? '+' : ''}${delta}`, 'difference'),
    ]),
    el('p', {}, [
      harsher
        ? 'The rubric is harsher than the crowd here. The items below are what '
          + 'the crowd either did not test or did not mind.'
        : 'The rubric is kinder than the crowd here, worth reading as a case '
          + 'where the crowd punished something the brief never asked for.',
    ]),
    el('div', { class: 'small dim' }, [
      `ViralScore range across runs: ${fmt(cmp.viral_score_min)}–`
      + `${fmt(cmp.viral_score_max)}. A gap smaller than that spread is not a finding.`,
    ]),
    el('h3', {}, [`What the rubric penalised (${failed.length})`]),
    failed.length
      ? el('table', { class: 'grid' }, [el('tbody', {}, failed.map((row) => el('tr', {
        style: 'cursor:pointer',
        onclick: () => { state.selected = row; state.tab = 'item'; renderItems(); renderTabs(); },
      }, [
        el('td', { class: 'mono' }, [row.id]),
        el('td', {}, [truncate(row.text, 80)]),
        el('td', { class: 'mono nowrap err-text' }, [
          row.kind === 'penalty' ? String(row.points) : `-${row.points}`,
        ]),
      ])))])
      : el('div', { class: 'small dim' }, ['Nothing. A clean sheet.']),
  ]);
}

function viewTranscript() {
  if (!state.grade.has_transcript) {
    return empty('No transcript was recorded for this grade.');
  }
  if (state.transcript === null) {
    // Lazy: the log is large and most visits never open this tab.
    api(`/api/rubric/${encodeURIComponent(state.runId)}/transcript?limit=400`)
      .then((data) => { state.transcript = data; if (state.tab === 'transcript') renderTabBody(); })
      .catch((err) => toast(String(err.message || err), true));
    return el('div', { class: 'empty' }, [el('span', { class: 'spin', text: '◐' }), ' loading…']);
  }
  const { calls, total } = state.transcript;
  return el('div', { class: 'feed' }, [
    el('div', { class: 'small dim', style: 'padding:6px 10px' }, [
      `${calls.length} of ${total} recorded tool calls. This is the harness's own `
      + 'log, written before the model saw each result.',
    ]),
    ...calls.map((call) => el('div', { class: 'call' }, [
      el('div', { class: 'call-head' }, [
        el('span', { class: 'mono accent' }, [call.id]),
        el('span', { class: 'badge' }, [call.name]),
        call.item_id && chip(call.item_id),
        !call.ok && chip('failed', 'err'),
      ].filter(Boolean)),
      Object.keys(call.args || {}).length
        ? el('div', { class: 'call-args' }, [JSON.stringify(call.args)])
        : null,
      el('pre', { class: 'block mono small' }, [
        call.result + (call.truncated ? '\n… [truncated]' : ''),
      ]),
    ].filter(Boolean))),
  ]);
}

function viewRubric() {
  return el('div', { class: 'detail' }, [
    el('div', { class: 'small dim' }, [
      'The rubric exactly as graded. Recorded in the grade itself, so this stays '
      + 'accurate even after the rubric file is edited.',
    ]),
    ...(state.grade.tiers || []).map((tier) => fold(
      `Tier ${tier.tier} · ${tier.label} (${tier.earned}/${tier.points})`,
      el('div', {}, (tier.items || []).map((item) => el('div', {
        style: 'margin-bottom:8px',
      }, [
        el('div', {}, [
          el('span', { class: 'mono' }, [`${item.id} `]),
          el('span', { class: 'mono faint' }, [`${item.points}p `]),
          chip(item.method, METHOD_CLASS[item.method] || ''),
        ]),
        el('div', { class: 'small' }, [item.text]),
        item.expect && el('div', { class: 'small dim' }, [`expected: ${item.expect}`]),
      ].filter(Boolean)))),
      tier.earned < tier.points,
    )),
    fold('Penalties', el('div', {}, (state.grade.penalties || []).map((p) => el('div', {
      class: 'small', style: 'margin-bottom:6px',
    }, [
      el('span', { class: 'mono' }, [`${p.id} `]),
      el('span', { class: 'mono err-text' }, [`${p.points} `]),
      p.text,
      p.fired && el('span', { class: 'chip err', style: 'margin-left:6px' }, ['fired']),
    ].filter(Boolean))))),
  ]);
}

function viewPaths() {
  const p = state.grade.paths || {};
  return el('div', { class: 'detail' }, [
    kv(Object.entries(p).map(([k, v]) => [k, el('code', { class: 'mono' }, [String(v)])])),
    (state.grade.shots || []).length ? el('div', {}, [
      el('h3', {}, [`Screenshots (${state.grade.shots.length})`]),
      el('div', { class: 'shot-grid' }, state.grade.shots.map((name) => el('img', {
        class: 'shot', loading: 'lazy',
        src: `/api/rubric/${encodeURIComponent(state.runId)}/shot/${encodeURIComponent(name)}`,
      }))),
    ]) : null,
  ].filter(Boolean));
}

// ---------------------------------------------------------------- browser

let browseTimer = null;

function openBrowser(prefill = '') {
  $('#browser').hidden = false;
  if (prefill) $('#searchInput').value = prefill;
  $('#searchInput').focus();
  refreshBrowse();
}

async function refreshBrowse() {
  const query = new URLSearchParams({ limit: '400' });
  const needle = $('#searchInput').value.trim();
  if (needle) query.set('q', needle);
  if ($('#divergedOnly').checked) query.set('diverged', '1');
  if ($('#gatedOnly').checked) query.set('gated', '1');
  const body = clear($('#browseBody'));
  try {
    const data = await api(`/api/rubric/grades?${query}`);
    if (!data.grades.length) { body.append(empty('No grades match.')); return; }
    body.append(el('div', { class: 'small dim', style: 'margin-bottom:6px' }, [
      `${data.matched} grade(s)`,
    ]));
    body.append(el('table', { class: 'grid' }, [
      el('thead', {}, [el('tr', {}, [
        el('th', {}, ['grade']), el('th', {}, ['idea']), el('th', {}, ['arm']),
        el('th', {}, ['model']), el('th', {}, ['rubric']), el('th', {}, ['viral']),
        el('th', {}, ['Δ']),
      ])]),
      el('tbody', {}, data.grades.map((row) => el('tr', {
        style: 'cursor:pointer',
        onclick: () => { $('#browser').hidden = true; load(row.run_id); },
      }, [
        el('td', { class: 'mono small' }, [row.run_id]),
        el('td', {}, [row.idea_id]),
        el('td', {}, [chip(row.arm || '?', ARM_CLASS[row.arm] || '')]),
        el('td', { class: 'small' }, [row.model_short || '']),
        el('td', { class: 'mono' }, [
          row.gate_zeroed
            ? el('span', { class: 'err-text' }, ['0 (gate)'])
            : fmt(row.score),
        ]),
        el('td', { class: 'mono' }, [fmt(row.viral_score)]),
        el('td', { class: 'mono' }, [
          row.delta === null || row.delta === undefined
            ? el('span', { class: 'faint' }, ['—'])
            : el('span', { class: Math.abs(row.delta) >= 20 ? 'warn-text' : '' },
              [`${row.delta > 0 ? '+' : ''}${row.delta}`]),
        ]),
      ]))),
    ]));
  } catch (err) { body.append(el('div', { class: 'empty err-text' }, [String(err.message || err)])); }
}

// ---------------------------------------------------------------- helpers

function fmt(value) {
  return value === null || value === undefined ? '—' : String(value);
}

function pct(value) {
  return value === null || value === undefined ? '—' : `${Math.round(value * 100)}%`;
}

function truncate(text, max) {
  const value = String(text || '');
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}

// ---------------------------------------------------------------- wiring

$('#loadBtn').onclick = () => load($('#gradeInput').value.trim());
$('#gradeInput').onkeydown = (event) => {
  if (event.key === 'Enter') load($('#gradeInput').value.trim());
};
$('#browseBtn').onclick = () => openBrowser();
$('#closeBrowse').onclick = () => { $('#browser').hidden = true; };
$('#searchInput').oninput = () => {
  clearTimeout(browseTimer);
  browseTimer = setTimeout(refreshBrowse, 220);
};
$('#divergedOnly').onchange = refreshBrowse;
$('#gatedOnly').onchange = refreshBrowse;
$('#dlJson').onclick = () => {
  location.href = `/api/rubric/${encodeURIComponent(state.runId)}/download`;
};
$('#dlZip').onclick = () => {
  location.href = `/api/rubric/${encodeURIComponent(state.runId)}/download?format=zip`;
};
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') $('#browser').hidden = true;
});

const initial = params.get('grade') || params.get('build');
if (initial) load(initial); else openBrowser();
