"use strict";
// draws one subject's tokenized timeline from the payload visualizer.py embeds
(function () {
  const D = JSON.parse(document.getElementById("cocoa-data").textContent);
  const app = document.getElementById("app");
  const css = getComputedStyle(document.documentElement);
  const token = (name) => css.getPropertyValue(name).trim();
  const INK = token("--ink");
  const INK2 = token("--ink-2");
  const MUTED = token("--muted");
  const SURFACE = token("--surface");
  const GRID = token("--grid");
  const GRID_SOFT = token("--grid-soft");
  const AXIS = token("--axis");
  const FONT = css.fontFamily;

  const LANE_H = 30; // a lane, with every code on one line
  const CODE_H = 24; // one code of an opened lane
  const BIN_H = 44; // one binned code, drawn at the height of its quantile
  const AXIS_H = 52;
  const OV_H = 40;
  const HIT = 12; // px either side of a mark that still counts as on it
  const MIN_SPAN = 60e3;
  const TOK_CAP = 2500;
  const ROW_CAP = 2000;
  const MIN = 6e4;
  const HR = 36e5;
  const DAY = 864e5;
  const MON = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(" ");
  const STEPS = [1e3, 5e3, 15e3, 3e4, MIN, 5 * MIN, 15 * MIN, 30 * MIN, HR, 2 * HR, 3 * HR]
    .concat([6 * HR, 12 * HR, DAY, 2 * DAY, 7 * DAY, 14 * DAY])
    .map((ms) => ({ ms, approx: ms }))
    .concat([1, 3, 6, 12, 24, 60, 120].map((mo) => ({ mo, approx: mo * 30.44 * DAY })));
  const nf = new Intl.NumberFormat("en-US");

  const E = D.events;
  const T = D.times.ms;
  const C = D.codes;
  const L = D.lanes;
  const TE = D.tokens.event; // token -> event
  const nE = E.kind ? E.kind.length : 0;
  const nT = D.tokens.id.length;
  const sub = D.subject;
  // an event, as counted, is an instant the data has something at; clocks are
  // synthetic, and the other structural tokens share their neighbors' times
  const stampIdx = [];
  for (let e = 0; e < nE; e++)
    if ((E.kind[e] === "event" || E.kind[e] === "unk") && E.time[e] !== stampIdx.at(-1))
      stampIdx.push(E.time[e]);
  const stamps = Float64Array.from(stampIdx, (i) => T[i]);
  const nStamps = stamps.length;

  app.append(header());
  if (!nE) {
    app.append(el("p", "empty", "This subject's timeline has no events."));
    return;
  }

  // events and tokens are in timeline order, so their times are sorted
  const ems = Float64Array.from(E.time, (i) => T[i]);
  const tms = Float64Array.from(TE, (e) => ems[e]);
  const W8 = D.winnowed;
  const lv = W8 ? W8.last_valid : null;
  const thr = lv != null && nT ? tms[Math.max(0, Math.min(nT, lv) - 1)] : null;
  const ends = [];
  for (let e = 0; e < nE; e++)
    if (E.kind[e] === "bos" || E.kind[e] === "eos") ends.push(e);

  const codeEv = C.map(() => []);
  const laneEv = L.map(() => []);
  for (let e = 0; e < nE; e++) {
    const c = E.code[e];
    codeEv[c].push(e);
    if (C[c].lane != null) laneEv[C[c].lane].push(e);
  }
  const laneCodes = L.map(() => []);
  C.forEach((c, i) => c.lane != null && laneCodes[c.lane].push(i));
  laneCodes.forEach((cs) =>
    cs.sort((a, b) => codeEv[b].length - codeEv[a].length || codeEv[a][0] - codeEv[b][0])
  );
  let maxBin = 0;
  const binned = codeEv.map((ev) => {
    let any = false;
    for (const e of ev)
      if (E.bin[e] != null) {
        any = true;
        maxBin = Math.max(maxBin, E.bin[e]);
      }
    return any;
  });
  const nBins = Math.max(2, D.meta.n_bins || maxBin + 1);
  const multi = L.map((lane) => lane.prefixes.length > 1);
  const hay = C.map((c) =>
    [c.code, c.name, c.category, c.description, c.lane != null ? L[c.lane].name : ""]
      .filter(Boolean)
      .join("\n")
      .toLowerCase()
  );
  const nav = []; // events a keyboard can step through
  for (let e = 0; e < nE; e++) if (C[E.code[e]].lane != null) nav.push(e);

  let ext0 = ems[0];
  let ext1 = ems[nE - 1];
  const pad = ext1 - ext0 < MIN ? 30 * MIN : (ext1 - ext0) * 0.015;
  ext0 -= pad;
  ext1 += pad;

  // wall-clock fields in the data's own zone, whatever the viewer's zone is
  let zone = D.meta.timezone || "UTC";
  let dtf;
  try {
    dtf = zoned(zone);
  } catch (err) {
    zone = "UTC";
    dtf = zoned(zone);
  }

  const S = {
    v0: ext0,
    v1: ext1,
    open: new Set(),
    match: null,
    nMatch: 0,
    hov: [],
    sel: -1,
    tab: "tokens",
    ticks: [],
    labels: [],
  };
  let rows = [];
  let laneRow = [];
  let codeRow = new Map();
  let totalH = 1;
  let W = 600;
  let mainScale = 1;
  let sized = false;

  // ---- page scaffold ----------------------------------------------------------

  const viz = el("section", "viz");
  const sticky = el("div", "sticky");
  const bar = el("div", "toolbar");
  const expandBtn = button("Open all lanes", "Show every code on its own row", () => {
    if (S.open.size === L.length) S.open.clear();
    else L.forEach((_, j) => S.open.add(j));
    layout();
  });
  const resetBtn = button("Reset", "Show the whole timeline again (0)", () =>
    setView(ext0, ext1)
  );
  const zoomOutBtn = button("−", "Zoom out (−)", () => zoomAt(W / 2, 2));
  const zoomInBtn = button("+", "Zoom in (+)", () => zoomAt(W / 2, 0.5));
  const zoom = el("div", "seg");
  zoom.setAttribute("role", "group");
  zoom.setAttribute("aria-label", "Zoom");
  zoom.append(zoomOutBtn, resetBtn, zoomInBtn);
  const search = el("input");
  search.type = "search";
  search.placeholder = "Highlight codes, e.g. VTL//heart_rate";
  search.setAttribute("aria-label", "Highlight codes");
  const searchCount = el("span", "count num");
  const glass = el("span", "glass");
  // parsed as html, so the svg needs no namespace url
  glass.innerHTML =
    '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="4.75"/>' +
    '<path d="m10.5 10.5 3.5 3.5"/></svg>';
  const slash = el("kbd", null, "/");
  slash.setAttribute("aria-hidden", "true");
  const field = el("span", "field");
  field.append(glass, search, slash);
  const searchBox = el("label", "search");
  searchBox.append(field, searchCount);
  const readout = el("span", "readout num");
  bar.append(zoom, expandBtn, searchBox, readout);
  const ovRow = el("div", "row ov-row");
  const ovCanvas = el("canvas");
  ovCanvas.setAttribute("aria-hidden", "true");
  ovRow.append(el("div", "gut", "Whole timeline"), cell(ovCanvas));
  const axisRow = el("div", "row");
  const axisCanvas = el("canvas");
  axisCanvas.setAttribute("aria-hidden", "true");
  axisRow.append(el("div", "gut", `Times in ${zone}`), cell(axisCanvas));
  sticky.append(bar, ovRow, axisRow);

  const bodyRow = el("div", "row");
  const labelsEl = el("div", "labels");
  const wrap = el("div", "cell plotwrap");
  wrap.tabIndex = 0;
  wrap.setAttribute("role", "group");
  wrap.setAttribute("aria-roledescription", "timeline");
  wrap.setAttribute(
    "aria-label",
    `Timeline of subject ${sub.subject_id}: ${nStamps} events in ${L.length} lanes. ` +
      "Arrow keys step through events; the table below lists every value."
  );
  const mainCanvas = el("canvas");
  const overCanvas = el("canvas");
  wrap.append(mainCanvas, overCanvas);
  bodyRow.append(labelsEl, wrap);
  const hint = el("div", "hint");
  for (const s of [
    "Drag to pan",
    "Pinch or [⌘]/[Ctrl]-scroll to zoom",
    "Double-click to zoom in",
    "Reset or [0] to see it all",
    "Click a lane to show each code",
    "Click an event, then [←] [→] to step",
  ])
    hint.append(withKeys(s));
  viz.append(sticky, bodyRow, hint);
  app.append(viz);

  const panel = el("section", "panel");
  const tabs = el("div", "tabs");
  tabs.setAttribute("role", "tablist");
  const tabBtns = {};
  for (const [key, label] of [
    ["tokens", "Tokens"],
    ["table", "Table"],
  ]) {
    const b = el("button", "tab", label);
    b.type = "button";
    b.setAttribute("role", "tab");
    b.addEventListener("click", () => {
      S.tab = key;
      renderPanel();
    });
    tabBtns[key] = b;
    tabs.append(b);
  }
  const panelNote = el("span", "panel-note num");
  tabs.append(panelNote);
  const panelBody = el("div", "panel-body");
  panelBody.setAttribute("role", "tabpanel");
  panel.append(tabs, panelBody);
  app.append(panel);
  if (W8) app.append(winnowCard());
  app.append(footer());

  const tip = el("div", "tip");
  tip.setAttribute("role", "tooltip");
  tip.hidden = true;
  document.body.append(tip);

  // ---- layout -----------------------------------------------------------------

  function layout() {
    rows = [];
    laneRow = [];
    codeRow = new Map();
    let y = 0;
    L.forEach((_, j) => {
      const r = { lane: j, code: -1, ev: laneEv[j], y, h: LANE_H };
      rows.push(r);
      laneRow[j] = r;
      y += LANE_H;
      if (S.open.has(j))
        for (const c of laneCodes[j]) {
          const h = binned[c] ? BIN_H : CODE_H;
          const cr = { lane: j, code: c, ev: codeEv[c], y, h };
          rows.push(cr);
          codeRow.set(c, cr);
          y += h;
        }
    });
    totalH = Math.max(y, 1);
    expandBtn.textContent = S.open.size === L.length ? "Close all lanes" : "Open all lanes";
    renderLabels();
    sizeMain();
    redraw(ALL);
  }

  function renderLabels() {
    const focused = document.activeElement?.dataset?.lane;
    labelsEl.replaceChildren();
    labelsEl.style.height = totalH + "px";
    for (const r of rows) {
      let n;
      if (r.code < 0) {
        const lane = L[r.lane];
        const open = S.open.has(r.lane);
        n = el("button", "lane");
        n.type = "button";
        n.dataset.lane = r.lane;
        n.setAttribute("aria-expanded", String(open));
        n.title = `${lane.prefixes.join(", ")} · click to ${open ? "close" : "show each code"}`;
        const sw = el("span", "sw");
        sw.style.background = lane.color;
        n.append(
          el("span", "chev", open ? "▾" : "▸"),
          sw,
          el("span", "nm", lane.name),
          el("span", "ct", fmtInt(r.ev.length))
        );
        n.addEventListener("click", () => {
          if (S.open.has(r.lane)) S.open.delete(r.lane);
          else S.open.add(r.lane);
          layout();
        });
      } else {
        const c = C[r.code];
        n = el("div", binned[r.code] ? "code binned" : "code");
        n.title = c.code + (c.description ? ` · ${c.description}` : "");
        if (multi[r.lane]) n.append(el("span", "pfx", c.prefix));
        n.append(el("span", "nm", c.name), el("span", "ct", fmtInt(r.ev.length)));
        if (binned[r.code])
          n.append(el("span", "qhi", `Q${nBins - 1}`), el("span", "qlo", "Q0"));
      }
      n.style.top = r.y + "px";
      n.style.height = r.h + "px";
      labelsEl.append(n);
    }
    if (focused != null) labelsEl.querySelector(`[data-lane="${focused}"]`)?.focus();
  }

  function sizeMain() {
    // browsers cap a canvas side near 32k device px; a tall canvas gets coarser
    mainScale = Math.min(devicePixelRatio || 1, 32000 / totalH);
    fit(mainCanvas, W, totalH, mainScale);
    fit(overCanvas, W, totalH, mainScale);
    wrap.style.height = totalH + "px";
  }

  function resize() {
    const w = Math.max(200, Math.floor(wrap.clientWidth));
    if (w === W && sized) return;
    W = w;
    sized = true;
    fit(axisCanvas, W, AXIS_H);
    fit(ovCanvas, W, OV_H);
    sizeMain();
    redraw(ALL);
  }

  // ---- drawing ----------------------------------------------------------------

  const MAIN = 1;
  const OVER = 2;
  const AXIS_F = 4;
  const OVV = 8;
  const ALL = 15;
  let dirty = 0;
  function redraw(flags) {
    if (!dirty) requestAnimationFrame(flush);
    dirty |= flags;
  }
  function flush() {
    const f = dirty;
    dirty = 0;
    if (f & (AXIS_F | MAIN)) makeTicks();
    if (f & AXIS_F) drawAxis();
    if (f & MAIN) drawMain();
    if (f & OVER) drawOverlay();
    if (f & OVV) drawOverview();
  }

  const X = (t) => ((t - S.v0) / (S.v1 - S.v0)) * W;
  const Tx = (x) => S.v0 + (x / W) * (S.v1 - S.v0);
  const dim = (e) => S.match !== null && !S.match[E.code[e]];
  const yBin = (r, b) => r.y + r.h - 7 - (b / (nBins - 1)) * (r.h - 14);

  function drawMain() {
    const ctx = ctx2d(mainCanvas, mainScale);
    ctx.fillStyle = SURFACE;
    ctx.fillRect(0, 0, W, totalH);
    for (const r of rows)
      if (r.code >= 0) {
        ctx.fillStyle = "rgba(0,0,0,0.018)";
        ctx.fillRect(0, r.y, W, r.h);
      }
    if (thr != null && X(thr) < W) {
      const x = Math.max(0, X(thr));
      ctx.fillStyle = "rgba(0,0,0,0.035)";
      ctx.fillRect(x, 0, W - x, totalH);
    }
    ctx.fillStyle = GRID_SOFT;
    for (const t of S.ticks) ctx.fillRect(Math.round(X(t)), 0, 1, totalH);
    for (const r of rows) {
      ctx.fillStyle = r.code < 0 ? GRID : GRID_SOFT;
      ctx.fillRect(0, r.y, W, 1);
    }
    ctx.fillStyle = AXIS;
    for (const e of ends) ctx.fillRect(Math.round(X(ems[e])), 0, 1, totalH);
    if (thr != null) {
      ctx.fillStyle = INK2;
      ctx.fillRect(Math.round(X(thr)), 0, 1, totalH);
    }
    for (const r of rows) {
      if (r.code >= 0 && binned[r.code]) drawBinned(ctx, r);
      else drawTicks(ctx, r);
    }
    ctx.globalAlpha = 1;
  }

  function visible(r) {
    const slack = (HIT / W) * (S.v1 - S.v0);
    return [lbEv(r.ev, S.v0 - slack), ubEv(r.ev, S.v1 + slack)];
  }

  function drawTicks(ctx, r) {
    const [a, b] = visible(r);
    const top = r.y + (r.code < 0 ? 8 : 6);
    const bot = r.y + r.h - (r.code < 0 ? 8 : 6);
    ctx.strokeStyle = L[r.lane].color;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    for (const faded of S.match ? [true, false] : [false]) {
      ctx.globalAlpha = faded ? 0.15 : 1;
      ctx.beginPath();
      let last = NaN;
      for (let k = a; k < b; k++) {
        const e = r.ev[k];
        if (S.match && dim(e) !== faded) continue;
        const px = Math.round(X(ems[e]) * mainScale); // one tick per device px
        if (px === last) continue;
        last = px;
        ctx.moveTo(px / mainScale, top);
        ctx.lineTo(px / mainScale, bot);
      }
      ctx.stroke();
    }
  }

  function drawBinned(ctx, r) {
    const [a, b] = visible(r);
    const color = L[r.lane].color;
    ctx.fillStyle = GRID_SOFT;
    ctx.fillRect(0, Math.round(yBin(r, nBins - 1)), W, 1);
    ctx.fillRect(0, Math.round(yBin(r, 0)), W, 1);
    ctx.globalAlpha = dim(r.ev[0]) ? 0.15 : 1;
    ctx.strokeStyle = ctx.fillStyle = color;
    ctx.lineCap = ctx.lineJoin = "round";
    let n = 0;
    for (let k = a; k < b; k++) if (E.bin[r.ev[k]] != null) n++;
    if (n > W / 2) {
      // too dense for a line: the range of quantiles per device px
      ctx.lineWidth = 2;
      ctx.beginPath();
      let px = NaN;
      let lo = 0;
      let hi = 0;
      const flushCol = () => {
        if (px !== px) return;
        ctx.moveTo(px / mainScale, yBin(r, lo));
        ctx.lineTo(px / mainScale, yBin(r, hi) - 0.01);
      };
      for (let k = a; k < b; k++) {
        const bin = E.bin[r.ev[k]];
        if (bin == null) continue;
        const p = Math.round(X(ems[r.ev[k]]) * mainScale);
        if (p !== px) {
          flushCol();
          px = p;
          lo = hi = bin;
        } else {
          lo = Math.min(lo, bin);
          hi = Math.max(hi, bin);
        }
      }
      flushCol();
      ctx.stroke();
    } else {
      const saved = ctx.globalAlpha;
      ctx.globalAlpha = saved * 0.45;
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      for (let k = a; k < b; k++) {
        const e = r.ev[k];
        if (E.bin[e] == null) continue;
        const x = X(ems[e]);
        const y = yBin(r, E.bin[e]);
        if (started) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
        started = true;
      }
      ctx.stroke();
      ctx.globalAlpha = saved;
      const big = n <= W / 8;
      for (let k = a; k < b; k++) {
        const e = r.ev[k];
        if (E.bin[e] == null) continue;
        const x = X(ems[e]);
        const y = yBin(r, E.bin[e]);
        if (big) {
          ctx.fillStyle = SURFACE;
          ctx.beginPath();
          ctx.arc(x, y, 5.5, 0, 2 * Math.PI);
          ctx.fill();
          ctx.fillStyle = color;
        }
        ctx.beginPath();
        ctx.arc(x, y, big ? 4 : 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    // a text value or unparseable number on a binned code has no height
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let k = a; k < b; k++)
      if (E.bin[r.ev[k]] == null) {
        const x = X(ems[r.ev[k]]);
        ctx.moveTo(x, r.y + r.h / 2 - 5);
        ctx.lineTo(x, r.y + r.h / 2 + 5);
      }
    ctx.stroke();
  }

  function rowsOf(e) {
    const c = E.code[e];
    const l = C[c].lane;
    if (l == null) return [];
    return codeRow.has(c) ? [laneRow[l], codeRow.get(c)] : [laneRow[l]];
  }

  function drawOverlay() {
    const ctx = ctx2d(overCanvas, mainScale);
    ctx.clearRect(0, 0, W, totalH);
    const marks = [...S.hov.map((e) => [e, false]), ...(S.sel >= 0 ? [[S.sel, true]] : [])];
    for (const [e, sel] of marks) {
      const x = X(ems[e]);
      if (x < -HIT || x > W + HIT) continue;
      ctx.fillStyle = "rgba(0,0,0,0.28)";
      ctx.fillRect(Math.round(x), 0, 1, totalH);
      ctx.strokeStyle = sel ? INK : INK2;
      ctx.lineWidth = 2;
      for (const r of rowsOf(e)) {
        ctx.beginPath();
        if (r.code >= 0 && binned[r.code] && E.bin[e] != null) {
          ctx.arc(x, yBin(r, E.bin[e]), 7, 0, 2 * Math.PI);
        } else {
          const inset = r.code < 0 ? 5 : 3;
          ctx.roundRect(x - 4, r.y + inset, 8, r.h - 2 * inset, 4);
        }
        ctx.stroke();
      }
    }
  }

  function makeTicks() {
    const span = S.v1 - S.v0;
    const want = span / Math.max(1, W / 110);
    const st = STEPS.find((s) => s.approx >= want) || STEPS[STEPS.length - 1];
    const out = [];
    if (st.mo) {
      const w = wall(S.v0);
      const m0 = Math.floor((w.y * 12 + w.mo - 1) / st.mo) * st.mo;
      for (let k = 0; k < 500; k++) {
        const m = m0 + k * st.mo;
        const t = fromWall(Date.UTC(Math.floor(m / 12), m % 12, 1));
        if (t > S.v1) break;
        if (t >= S.v0) out.push(t);
      }
    } else {
      for (let w = Math.floor(wallMs(S.v0) / st.ms) * st.ms; out.length < 500; w += st.ms) {
        const t = fromWall(w);
        if (t > S.v1) break;
        if (t >= S.v0 && t !== out[out.length - 1]) out.push(t);
      }
    }
    S.ticks = out;
    S.labels = tickLabels(out, st);
  }

  function tickLabels(ticks, st) {
    let prev = null;
    return ticks.map((t) => {
      const w = wall(t);
      const newYear = !prev || prev.y !== w.y;
      const newDay = newYear || prev.mo !== w.mo || prev.d !== w.d;
      let a;
      let b = "";
      if (!st.mo && st.ms < DAY) {
        a = `${p2(w.h)}:${p2(w.mi)}` + (st.ms < MIN ? `:${p2(w.s)}` : "");
        if (newDay) b = `${MON[w.mo - 1]} ${w.d}` + (newYear ? `, ${w.y}` : "");
      } else if (!st.mo) {
        a = `${MON[w.mo - 1]} ${w.d}`;
        if (newYear) b = String(w.y);
      } else if (st.mo < 12) {
        a = MON[w.mo - 1];
        if (newYear) b = String(w.y);
      } else a = String(w.y);
      prev = w;
      return [a, b];
    });
  }

  function drawAxis() {
    const ctx = ctx2d(axisCanvas);
    ctx.clearRect(0, 0, W, AXIS_H);
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "center";
    S.ticks.forEach((t, i) => {
      const x = X(t);
      const [a, b] = S.labels[i];
      ctx.fillStyle = AXIS;
      ctx.fillRect(Math.round(x), 30, 1, 5);
      ctx.fillStyle = INK2;
      ctx.font = `11px ${FONT}`;
      fillClamped(ctx, a, x, 13);
      if (b) {
        ctx.fillStyle = INK;
        ctx.font = `500 11px ${FONT}`;
        fillClamped(ctx, b, x, 26);
      }
    });
    ctx.fillStyle = AXIS;
    ctx.fillRect(0, AXIS_H - 1, W, 1);
    // markers: the winnowing threshold wins any overlap with BOS and EOS
    ctx.font = `11px ${FONT}`;
    const taken = [];
    const place = (text, x, align) => {
      const w = ctx.measureText(text).width;
      const x0 = align === "left" ? x : x - w;
      if (x0 < 0 || x0 + w > W || taken.some(([a, b]) => x0 < b + 6 && x0 + w > a - 6))
        return;
      taken.push([x0, x0 + w]);
      ctx.textAlign = align;
      ctx.fillText(text, x, 47);
    };
    if (thr != null) {
      const x = X(thr);
      if (x >= 0 && x <= W) {
        ctx.fillStyle = INK2;
        ctx.fillRect(Math.round(x), 37, 1, AXIS_H - 37);
        place("past", x - 5, "right");
        place("future", x + 5, "left");
      }
    }
    for (const e of ends) {
      const x = X(ems[e]);
      if (x < 0 || x > W) continue;
      ctx.fillStyle = AXIS;
      ctx.fillRect(Math.round(x), 37, 1, AXIS_H - 37);
      ctx.fillStyle = INK2;
      if (E.kind[e] === "bos") place("BOS", x + 4, "left");
      else place("EOS", x - 4, "right");
    }
  }

  const OX = (t) => ((t - ext0) / (ext1 - ext0)) * W;
  const OT = (x) => ext0 + (x / W) * (ext1 - ext0);

  function drawOverview() {
    const ctx = ctx2d(ovCanvas);
    const s = devicePixelRatio || 1;
    ctx.fillStyle = SURFACE;
    ctx.fillRect(0, 0, W, OV_H);
    const band = Math.max(2, Math.min(5, Math.floor((OV_H - 10) / Math.max(1, L.length))));
    const top = (OV_H - band * L.length) / 2;
    L.forEach((lane, j) => {
      ctx.fillStyle = lane.color;
      for (const faded of S.match ? [true, false] : [false]) {
        ctx.globalAlpha = faded ? 0.15 : 1;
        let last = NaN;
        for (const e of laneEv[j]) {
          if (S.match && dim(e) !== faded) continue;
          const px = Math.round(OX(ems[e]) * s);
          if (px === last) continue;
          last = px;
          ctx.fillRect(px / s, top + j * band, 1.5, band - 1);
        }
      }
    });
    ctx.globalAlpha = 1;
    if (thr != null) {
      ctx.fillStyle = INK2;
      ctx.fillRect(Math.round(OX(thr)), 0, 1, OV_H);
    }
    const x0 = OX(S.v0);
    const x1 = OX(S.v1);
    ctx.fillStyle = "rgba(255,255,255,0.72)";
    ctx.fillRect(0, 0, Math.max(0, x0), OV_H);
    ctx.fillRect(x1, 0, W - x1, OV_H);
    ctx.strokeStyle = INK2;
    ctx.lineWidth = 1;
    ctx.strokeRect(Math.round(x0) + 0.5, 0.5, Math.max(1, Math.round(x1 - x0) - 1), OV_H - 1);
    ctx.fillStyle = INK2;
    for (const x of [x0, x1]) {
      ctx.beginPath();
      ctx.roundRect(Math.round(x) - 2, OV_H / 2 - 7, 4, 14, 2);
      ctx.fill();
    }
  }

  // ---- view -------------------------------------------------------------------

  function setView(a, b) {
    const full = ext1 - ext0;
    let span = Math.min(full, Math.max(MIN_SPAN, b - a));
    let v0 = (a + b) / 2 - span / 2;
    v0 = Math.min(Math.max(v0, ext0), ext1 - span);
    S.v0 = v0;
    S.v1 = v0 + span;
    // a button that has nothing left to do says so, rather than doing nothing
    const whole = span >= full * 0.999;
    if (whole && [resetBtn, zoomOutBtn].includes(document.activeElement)) wrap.focus();
    resetBtn.disabled = zoomOutBtn.disabled = whole;
    zoomInBtn.disabled = span <= MIN_SPAN * 1.001;
    redraw(ALL);
    readout.textContent =
      `${fmtWall(S.v0)} – ${fmtWall(S.v1)} · ` +
      `${fmtInt(ub(stamps, S.v1) - lb(stamps, S.v0))} of ${fmtInt(nStamps)} events in view`;
    schedulePanel();
  }

  function zoomAt(x, factor) {
    const t = Tx(x);
    const span = Math.min(ext1 - ext0, Math.max(MIN_SPAN, (S.v1 - S.v0) * factor));
    const a = t - (x / W) * span;
    setView(a, a + span);
  }

  function reveal(e) {
    const t = ems[e];
    const margin = (S.v1 - S.v0) * 0.05;
    if (t < S.v0 + margin || t > S.v1 - margin) {
      const span = S.v1 - S.v0;
      setView(t - span / 2, t + span / 2);
    }
  }

  // ---- interaction: plot ------------------------------------------------------

  function hitTest(x, y) {
    const r = rows.find((q) => y >= q.y && y < q.y + q.h);
    if (!r || !r.ev.length) return null;
    const k = lbEv(r.ev, Tx(x));
    let best = -1;
    let bd = Infinity;
    for (const j of [k - 1, k]) {
      if (j < 0 || j >= r.ev.length) continue;
      const d = Math.abs(X(ems[r.ev[j]]) - x);
      if (d < bd) {
        bd = d;
        best = j;
      }
    }
    if (best < 0 || bd > HIT) return null;
    // everything this row holds at that instant
    const ti = E.time[r.ev[best]];
    let a = best;
    let b = best + 1;
    while (a > 0 && E.time[r.ev[a - 1]] === ti) a--;
    while (b < r.ev.length && E.time[r.ev[b]] === ti) b++;
    return { row: r, events: Array.from(r.ev.slice(a, b)) };
  }

  const local = (ev) => {
    const b = wrap.getBoundingClientRect();
    return [ev.clientX - b.left, ev.clientY - b.top];
  };

  let drag = null;
  wrap.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    wrap.setPointerCapture(ev.pointerId);
    drag = { x: ev.clientX, v0: S.v0, v1: S.v1, moved: false };
  });
  wrap.addEventListener("pointermove", (ev) => {
    if (drag) {
      const dx = ev.clientX - drag.x;
      if (Math.abs(dx) > 3) drag.moved = true;
      if (drag.moved) {
        wrap.classList.add("dragging");
        const dt = (-dx / W) * (drag.v1 - drag.v0);
        setView(drag.v0 + dt, drag.v1 + dt);
        setHover([]);
        return;
      }
    }
    const [x, y] = local(ev);
    const h = hitTest(x, y);
    wrap.classList.toggle("on-mark", !!h);
    setHover(h ? h.events : [], h ? [ev.clientX, ev.clientY] : null);
  });
  wrap.addEventListener("pointerup", (ev) => {
    wrap.classList.remove("dragging");
    if (drag && !drag.moved) {
      const [x, y] = local(ev);
      const h = hitTest(x, y);
      select(h ? h.events[0] : -1);
    }
    drag = null;
  });
  wrap.addEventListener("pointerleave", () => {
    if (!drag) setHover([]);
  });
  wrap.addEventListener("dblclick", (ev) => {
    zoomAt(local(ev)[0], ev.shiftKey ? 2 : 0.5);
  });
  wrap.addEventListener(
    "wheel",
    (ev) => {
      const unit = ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? W : 1;
      const dx = ev.deltaX * unit;
      const dy = ev.deltaY * unit;
      if (ev.ctrlKey || ev.metaKey) {
        ev.preventDefault();
        zoomAt(local(ev)[0], Math.exp(Math.max(-60, Math.min(60, dy)) * 0.01));
      } else if (Math.abs(dx) > Math.abs(dy) || ev.shiftKey) {
        ev.preventDefault();
        const d = Math.abs(dx) > Math.abs(dy) ? dx : dy;
        const dt = (d / W) * (S.v1 - S.v0);
        setView(S.v0 + dt, S.v1 + dt);
      } // a plain vertical scroll still scrolls the page
    },
    { passive: false }
  );
  wrap.addEventListener("keydown", (ev) => {
    const k = ev.key;
    if (k === "ArrowRight" || k === "ArrowLeft") step(k === "ArrowRight" ? 1 : -1);
    else if (k === "Home" || k === "End") {
      const list = navList();
      if (list.length) select(list[k === "Home" ? 0 : list.length - 1], true);
    } else if (k === "+" || k === "=") zoomAt(W / 2, 0.5);
    else if (k === "-" || k === "_") zoomAt(W / 2, 2);
    else if (k === "0") setView(ext0, ext1);
    else if (k === "Escape") select(-1);
    else return;
    ev.preventDefault();
  });
  wrap.addEventListener("blur", () => {
    if (!S.hov.length) hideTip();
  });

  const navList = () => (S.match ? nav.filter((e) => S.match[E.code[e]]) : nav);

  function step(dir) {
    const list = navList();
    if (!list.length) return;
    let i;
    if (S.sel < 0) {
      const inView = (e) => ems[e] >= S.v0 && ems[e] <= S.v1;
      i = dir > 0 ? list.findIndex(inView) : list.findLastIndex(inView);
      if (i < 0) i = dir > 0 ? 0 : list.length - 1;
    } else {
      i = lb(list, S.sel);
      if (dir > 0) i = list[i] === S.sel ? i + 1 : i;
      else i -= 1;
      i = Math.max(0, Math.min(list.length - 1, i));
    }
    select(list[i], true);
  }

  function select(e, fromKeys = false) {
    S.sel = e;
    if (e >= 0 && fromKeys) reveal(e);
    redraw(OVER);
    markPanel();
    if (e >= 0 && fromKeys) {
      // keyboard users get the same details a pointer gets on hover
      requestAnimationFrame(() => {
        const r = rowsOf(e).at(-1);
        const b = wrap.getBoundingClientRect();
        const y = r ? b.top + r.y + r.h / 2 : b.top;
        const group = r ? groupAt(r, e) : [e];
        showTip(group, b.left + X(ems[e]), y);
      });
      scrollPanelTo(e);
    } else if (e < 0) hideTip();
    else scrollPanelTo(e);
  }

  function groupAt(r, e) {
    // the selected event leads, then whatever else its row holds at that instant
    const ti = E.time[e];
    const out = [e];
    for (let k = lbEv(r.ev, ems[e]); k < r.ev.length && E.time[r.ev[k]] === ti; k++)
      if (r.ev[k] !== e) out.push(r.ev[k]);
    return out;
  }

  function setHover(events, at) {
    const same = events.length === S.hov.length && events.every((e, i) => e === S.hov[i]);
    S.hov = events;
    if (!same) {
      redraw(OVER);
      markPanel();
    }
    if (events.length && at) {
      if (same && !tip.hidden) placeTip(at[0], at[1]);
      else showTip(events, at[0], at[1]);
    } else if (!events.length) hideTip();
  }

  // ---- interaction: overview --------------------------------------------------

  let brush = null;
  const ovX = (ev) => ev.clientX - ovCanvas.getBoundingClientRect().left;
  function ovMode(x) {
    // a window over the whole timeline has nowhere to go, so any drag brushes
    if (S.v1 - S.v0 >= (ext1 - ext0) * 0.98) return "new";
    const x0 = OX(S.v0);
    const x1 = OX(S.v1);
    if (Math.abs(x - x0) < 7) return "l";
    if (Math.abs(x - x1) < 7) return "r";
    return x > x0 && x < x1 ? "move" : "new";
  }
  ovCanvas.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    ovCanvas.setPointerCapture(ev.pointerId);
    const x = ovX(ev);
    brush = { mode: ovMode(x), x, v0: S.v0, v1: S.v1, moved: false };
  });
  ovCanvas.addEventListener("pointermove", (ev) => {
    const x = ovX(ev);
    if (!brush) {
      const m = ovMode(x);
      ovCanvas.style.cursor = m === "new" ? "crosshair" : m === "move" ? "grab" : "ew-resize";
      return;
    }
    if (Math.abs(x - brush.x) > 3) brush.moved = true;
    if (!brush.moved) return;
    const t = OT(x);
    if (brush.mode === "move") {
      const dt = ((x - brush.x) / W) * (ext1 - ext0);
      setView(brush.v0 + dt, brush.v1 + dt);
    } else if (brush.mode === "l") setView(Math.min(t, brush.v1 - MIN_SPAN), brush.v1);
    else if (brush.mode === "r") setView(brush.v0, Math.max(t, brush.v0 + MIN_SPAN));
    else {
      const t0 = OT(brush.x);
      setView(Math.min(t0, t), Math.max(t0, t));
    }
  });
  ovCanvas.addEventListener("pointerup", (ev) => {
    if (brush && !brush.moved && brush.mode === "new") {
      const t = OT(ovX(ev));
      const span = S.v1 - S.v0;
      setView(t - span / 2, t + span / 2);
    }
    brush = null;
  });
  ovCanvas.addEventListener("dblclick", () => setView(ext0, ext1));

  // ---- interaction: search ----------------------------------------------------

  let searchTimer = 0;
  search.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => setQuery(search.value), 120);
  });
  search.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      setQuery(search.value);
      step(ev.shiftKey ? -1 : 1);
      ev.preventDefault();
    } else if (ev.key === "Escape") {
      search.value = "";
      setQuery("");
    }
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "/" && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName)) {
      search.focus();
      search.select();
      ev.preventDefault();
    }
  });

  function setQuery(q) {
    q = q.trim().toLowerCase();
    if (!q) {
      S.match = null;
      searchCount.textContent = "";
    } else {
      S.match = new Uint8Array(C.length);
      hay.forEach((h, i) => (S.match[i] = h.includes(q) ? 1 : 0));
      let n = 0;
      for (let e = 0, last = -1; e < nE; e++)
        if (S.match[E.code[e]] && E.time[e] !== last) {
          n++;
          last = E.time[e];
        }
      searchCount.textContent = `${fmtInt(n)} matching event${n === 1 ? "" : "s"}`;
    }
    redraw(MAIN | OVV);
    schedulePanel();
  }

  // ---- tooltip ----------------------------------------------------------------

  function showTip(events, x, y) {
    tip.replaceChildren();
    const e0 = events[0];
    const since = sub.start_time ? sub.start_time.ms : T[0];
    const head = el("div", "tip-time num", D.times.label[E.time[e0]]);
    const h = E.hours_to_end[e0];
    const toEnd = h == null ? "" : h >= 0 ? ` · ${fmtNum(h)} h to end` : ` · ${fmtNum(-h)} h past end`;
    head.append(el("span", null, ` · ${fmtDur(ems[e0] - since)} from start${toEnd}`));
    tip.append(head);
    const shown = events.slice(0, 6);
    for (const e of shown) tip.append(tipEvent(e));
    if (events.length > shown.length)
      tip.append(el("div", "tip-more", `+${events.length - shown.length} more at this time`));
    if (lv != null && E.first_token[e0] >= lv)
      tip.append(el("div", "tip-future", "After the winnowing threshold (future)"));
    tip.hidden = false;
    placeTip(x, y);
  }

  function placeTip(x, y) {
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    let left = x + 14;
    if (left + w > innerWidth - 8) left = x - 14 - w;
    let top = y + 14;
    if (top + h > innerHeight - 8) top = y - 14 - h;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  }

  function hideTip() {
    tip.hidden = true;
  }

  function tipEvent(e) {
    const c = C[E.code[e]];
    const v = E.value[e];
    const b = E.bin[e];
    const text = E.text[e];
    const box = el("div", "tip-ev");
    const key = el("span", "key");
    key.style.background = c.lane != null ? L[c.lane].color : MUTED;
    const body = el("div");
    const line = el("div", "tip-head");
    let lead;
    if (v != null) lead = fmtNum(v, 6);
    else if (b != null) lead = `Q${b}`;
    else if (text != null) lead = text.replace(/_/g, " ");
    else if (E.kind[e] === "clock") lead = `${c.name}:00`;
    else lead = c.name || c.code;
    line.append(el("strong", "num", lead));
    const named = v != null || b != null || text != null;
    const label = [named ? c.name : null, c.category].filter(Boolean).join(" · ");
    if (label) line.append(el("span", null, label));
    body.append(line);
    if (c.description) body.append(el("div", "tip-desc", c.description));
    if (b != null) body.append(el("div", "tip-meta num", binRange(c, b)));
    if (text != null && (v != null || b != null)) body.append(el("div", "tip-meta", text));
    const toks = [];
    for (let i = E.first_token[e]; i < E.first_token[e] + E.n_tokens[e]; i++)
      toks.push(`#${i}  ${D.tokens.id[i]}  ${D.tokens.vocab[D.tokens.id[i]]}`);
    body.append(el("div", "tip-tok", toks.join("\n")));
    box.append(key, body);
    return box;
  }

  function binRange(c, b) {
    const q = `Q${b} of Q0–Q${nBins - 1}`;
    const br = c.breaks;
    if (!br) return q;
    const lo = b > 0 ? br[b - 1] : null;
    const hi = b < br.length ? br[b] : null;
    const range =
      lo == null ? `below ${fmtNum(hi)}` : hi == null ? `${fmtNum(lo)} and above` : `${fmtNum(lo)} to ${fmtNum(hi)}`;
    return `${q} · ${range} in training`;
  }

  // ---- tokens and table -------------------------------------------------------

  let panelTimer = 0;
  let chipsOf = new Map();
  function schedulePanel() {
    clearTimeout(panelTimer);
    panelTimer = setTimeout(renderPanel, 150);
  }

  function renderPanel() {
    for (const [k, b] of Object.entries(tabBtns)) b.setAttribute("aria-selected", String(k === S.tab));
    const keep = panelBody.scrollTop;
    chipsOf = new Map();
    if (S.tab === "tokens") renderTokens();
    else renderTable();
    panelBody.scrollTop = keep;
    markPanel();
    if (S.sel >= 0) scrollPanelTo(S.sel);
  }

  function capped(a, b, focus, cap) {
    if (b - a <= cap) return [a, b];
    let s = focus >= a && focus < b ? Math.max(a, focus - (cap >> 1)) : a;
    s = Math.min(s, b - cap);
    return [s, s + cap];
  }

  function renderTokens() {
    const a = lb(tms, S.v0);
    const b = ub(tms, S.v1);
    const [s, t] = capped(a, b, S.sel >= 0 ? E.first_token[S.sel] : -1, TOK_CAP);
    panelNote.textContent =
      b - a > t - s
        ? `Showing ${fmtInt(t - s)} of the ${fmtInt(b - a)} tokens in view ` +
          `(#${fmtInt(s)}–#${fmtInt(t - 1)}); zoom in for the rest`
        : `${fmtInt(b - a)} of ${fmtInt(nT)} tokens in view`;
    const frag = document.createDocumentFragment();
    let chips = null;
    let prevTi = -1;
    let prevDate = null;
    for (let i = s; i < t; i++) {
      const e = TE[i];
      const ti = E.time[e];
      if (ti !== prevTi) {
        prevTi = ti;
        const [date, time] = D.times.label[ti].split(" ");
        if (date !== prevDate) frag.append(el("div", "day num", date)); // a row of its own
        prevDate = date;
        const g = el("div", "tg");
        chips = el("div", "chips");
        g.append(el("div", "tm", time), chips);
        frag.append(g);
      }
      if (i === lv) {
        const d = el("span", "thr", "threshold");
        d.title = "Tokens from here on are the future that winnowing held out";
        chips.append(d);
      }
      const c = C[E.code[e]];
      const chip = el("span", "tok");
      chip.dataset.e = e;
      chip.style.setProperty("--c", c.lane != null ? L[c.lane].color : MUTED);
      if (dim(e)) chip.classList.add("dim");
      if (lv != null && i >= lv) chip.classList.add("future");
      chip.append(el("span", null, D.tokens.vocab[D.tokens.id[i]]), el("span", "id", D.tokens.id[i]));
      chips.append(chip);
      if (!chipsOf.has(e)) chipsOf.set(e, []);
      chipsOf.get(e).push(chip);
    }
    panelBody.replaceChildren(frag);
    if (t < b) panelBody.append(el("div", "more", `${fmtInt(b - t)} more tokens in view.`));
  }

  function renderTable() {
    const a = lb(ems, S.v0);
    const b = ub(ems, S.v1);
    const [s, t] = capped(a, b, S.sel, ROW_CAP);
    panelNote.textContent =
      b - a > t - s
        ? `Showing ${fmtInt(t - s)} of the ${fmtInt(b - a)} rows in view; zoom in for the rest`
        : `${fmtInt(b - a)} of ${fmtInt(nE)} rows in view`;
    const cols = [
      ["Token", "r", (e) => fmtInt(E.first_token[e])],
      ["Time", "num", (e) => D.times.label[E.time[e]]],
      ["Lane", null, null],
      ["Code", "mono", (e) => C[E.code[e]].code],
      ["Bin", "r num", (e) => (E.bin[e] != null ? `Q${E.bin[e]}` : "")],
    ];
    if (D.meta.has_numeric_values) cols.push(["Value", "r num", (e) => fmtNum(E.value[e], 6)]);
    cols.push(["Text", null, (e) => E.text[e] ?? ""]);
    cols.push([
      "Tokens",
      "mono",
      (e) => {
        const out = [];
        for (let i = E.first_token[e]; i < E.first_token[e] + E.n_tokens[e]; i++)
          out.push(`${D.tokens.id[i]} ${D.tokens.vocab[D.tokens.id[i]]}`);
        return out.join("  ");
      },
    ]);
    const table = el("table", "events");
    const hr = el("tr");
    for (const [name, cls] of cols) hr.append(el("th", cls, name));
    const thead = el("thead");
    thead.append(hr);
    table.append(thead);
    const tb = el("tbody");
    for (let e = s; e < t; e++) {
      const tr = el("tr");
      tr.dataset.e = e;
      if (dim(e)) tr.style.opacity = "0.45";
      for (const [name, cls, get] of cols) {
        if (name === "Lane") {
          const c = C[E.code[e]];
          const td = el("td");
          if (c.lane != null) {
            const k = el("span", "lane-key");
            k.style.background = L[c.lane].color;
            td.append(k, L[c.lane].name);
          }
          tr.append(td);
        } else tr.append(el("td", cls, get(e)));
      }
      tb.append(tr);
      chipsOf.set(e, [tr]);
    }
    table.append(tb);
    panelBody.replaceChildren(table);
  }

  function markPanel() {
    for (const n of panelBody.querySelectorAll(".hov, .sel")) n.classList.remove("hov", "sel");
    for (const e of S.hov) for (const n of chipsOf.get(e) || []) n.classList.add("hov");
    for (const n of chipsOf.get(S.sel) || []) n.classList.add("sel");
  }

  function scrollPanelTo(e) {
    const n = (chipsOf.get(e) || [])[0];
    if (!n) return;
    const pb = panelBody.getBoundingClientRect();
    const nb = n.getBoundingClientRect();
    if (nb.top < pb.top + 30 || nb.bottom > pb.bottom)
      panelBody.scrollTop += nb.top - pb.top - pb.height / 3;
  }

  const eventOf = (target) => {
    const n = target.closest?.("[data-e]");
    return n && panelBody.contains(n) ? [Number(n.dataset.e), n] : [null, null];
  };
  panelBody.addEventListener("pointerover", (ev) => {
    const [e, n] = eventOf(ev.target);
    if (e == null) return setHover([]);
    const r = n.getBoundingClientRect();
    setHover([e], [r.left, r.bottom - 10]);
  });
  panelBody.addEventListener("pointerleave", () => setHover([]));
  panelBody.addEventListener("click", (ev) => {
    const [e] = eventOf(ev.target);
    if (e != null) select(e);
  });

  // ---- header, winnowing summary, footer ----------------------------------------

  function header() {
    const hd = el("header", "hd");
    const title = el("div", "title");
    const icon = document.querySelector('link[rel="icon"]');
    if (icon) {
      const img = el("img", "logo");
      img.src = icon.href;
      img.alt = "";
      title.append(img);
    }
    title.append(el("h1", null, `Subject ${sub.subject_id}`));
    if (sub.split) title.append(el("span", "badge", sub.split));
    hd.append(title);
    const start = sub.start_time;
    const end = sub.end_time;
    const facts = [];
    if (start) facts.push(["Start", start.label]);
    if (end) facts.push(["End", end.label]);
    if (start && end) facts.push(["Span", fmtDur(end.ms - start.ms)]);
    facts.push(["Tokens", fmtInt(nT)], ["Events", fmtInt(nStamps)], ["Distinct codes", fmtInt(C.length)]);
    hd.append(definitions("facts num", facts));
    const fields = sub.fields.filter((f) => f.value != null).map((f) => [f.label, f.value]);
    if (fields.length) hd.append(definitions("fields", fields));
    return hd;
  }

  function winnowCard() {
    const card = el("section", "card");
    const hd = el("div", "card-hd");
    hd.append(el("h2", null, "Winnowed for inference"), el("span", "card-note", W8.file));
    card.append(hd);
    const nPast = Math.min(lv, nT);
    const facts = [];
    if (nPast > 0) facts.push(["Threshold", D.times.label[E.time[TE[nPast - 1]]]]);
    facts.push(["Past tokens", fmtInt(nPast)], ["Future tokens", fmtInt(nT - nPast)]);
    if (W8.n_future != null && W8.n_future !== nT - nPast)
      facts.push(["Within the horizon", fmtInt(W8.n_future)]);
    card.append(definitions("facts num", facts));
    // flagged outcomes get a row each; the rest share one line
    const flagged = W8.outcomes.filter((o) => o.past || o.future);
    const absent = W8.outcomes.filter((o) => !o.past && !o.future);
    if (flagged.length) {
      const list = el("div", "outcomes");
      for (const o of flagged) {
        const item = el("div", "outcome");
        item.append(outcomeBtn(o.code), flag(o.past, "past"), flag(o.future, "future"));
        list.append(item);
      }
      card.append(list);
    }
    if (absent.length) {
      const rest = el("p", "absent", "Absent from both: ");
      absent.forEach((o, i) => rest.append(i ? ", " : "", outcomeBtn(o.code)));
      card.append(rest);
    }
    return card;
  }

  function outcomeBtn(code) {
    const b = el("button", "linkish", code);
    b.type = "button";
    b.title = `Highlight ${code} in the timeline`;
    b.addEventListener("click", () => {
      search.value = code;
      setQuery(code);
      viz.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    return b;
  }

  function flag(v, tense) {
    const f = el("span", v ? "flag on" : "flag", tense);
    f.title = `${v ? "In" : "Not in"} the ${tense} tokens`;
    return f;
  }

  function footer() {
    const m = D.meta;
    const f = el("footer", "meta");
    for (const s of [
      `cocoa ${m.cocoa_version}`,
      `rendered ${m.generated}`,
      m.tokenizer_created ? `tokenizer created ${m.tokenizer_created}` : null,
      `vocabulary of ${fmtInt(m.vocab_size)}`,
      m.fused ? "fused bins" : "unfused bins",
      m.value_source === "meds" ? "values from meds.parquet" : null,
      m.n_bins ? `${m.n_bins} bins` : null,
      `times in ${m.timezone}, ${m.time_unit}`,
      m.processed_data_home,
    ])
      if (s) f.append(el("span", null, s));
    return f;
  }

  function definitions(cls, pairs) {
    const dl = el("dl", cls);
    for (const [k, v] of pairs) {
      const d = el("div");
      d.append(el("dt", null, k), el("dd", null, v));
      dl.append(d);
    }
    return dl;
  }

  // ---- helpers ----------------------------------------------------------------

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  // text with its [bracketed] parts set as keys
  function withKeys(s) {
    const n = el("span");
    s.split(/\[([^\]]+)\]/).forEach((part, i) => {
      if (part) n.append(i % 2 ? el("kbd", null, part) : part);
    });
    return n;
  }

  function button(label, title, fn) {
    const b = el("button", "btn", label);
    b.type = "button";
    b.title = title;
    b.setAttribute("aria-label", title);
    b.addEventListener("click", fn);
    return b;
  }

  function cell(child) {
    const c = el("div", "cell");
    c.append(child);
    return c;
  }

  function fit(canvas, w, h, scale = devicePixelRatio || 1) {
    canvas.width = Math.max(1, Math.round(w * scale));
    canvas.height = Math.max(1, Math.round(h * scale));
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
  }

  function ctx2d(canvas, scale = devicePixelRatio || 1) {
    const ctx = canvas.getContext("2d");
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.globalAlpha = 1;
    return ctx;
  }

  function fillClamped(ctx, s, x, y) {
    const w = ctx.measureText(s).width;
    ctx.fillText(s, Math.max(w / 2 + 2, Math.min(W - w / 2 - 2, x)), y);
  }

  function lb(arr, t) {
    let lo = 0;
    let hi = arr.length;
    while (lo < hi) {
      const m = (lo + hi) >>> 1;
      if (arr[m] < t) lo = m + 1;
      else hi = m;
    }
    return lo;
  }

  function ub(arr, t) {
    let lo = 0;
    let hi = arr.length;
    while (lo < hi) {
      const m = (lo + hi) >>> 1;
      if (arr[m] <= t) lo = m + 1;
      else hi = m;
    }
    return lo;
  }

  function lbEv(ev, t) {
    let lo = 0;
    let hi = ev.length;
    while (lo < hi) {
      const m = (lo + hi) >>> 1;
      if (ems[ev[m]] < t) lo = m + 1;
      else hi = m;
    }
    return lo;
  }

  function ubEv(ev, t) {
    let lo = 0;
    let hi = ev.length;
    while (lo < hi) {
      const m = (lo + hi) >>> 1;
      if (ems[ev[m]] <= t) lo = m + 1;
      else hi = m;
    }
    return lo;
  }

  function zoned(tz) {
    return new Intl.DateTimeFormat("en-US", {
      timeZone: tz,
      hourCycle: "h23",
      year: "numeric",
      month: "numeric",
      day: "numeric",
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
    });
  }

  function wall(ms) {
    const p = {};
    for (const { type, value } of dtf.formatToParts(ms)) p[type] = value;
    return { y: +p.year, mo: +p.month, d: +p.day, h: +p.hour % 24, mi: +p.minute, s: +p.second };
  }

  // an instant as the utc-labeled ms of its wall-clock reading, and back
  function wallMs(ms) {
    const w = wall(ms);
    return Date.UTC(w.y, w.mo - 1, w.d, w.h, w.mi, w.s) + (((ms % 1000) + 1000) % 1000);
  }

  function fromWall(wms) {
    let u = wms - (wallMs(wms) - wms);
    u = wms - (wallMs(u) - u);
    return u;
  }

  function fmtWall(ms) {
    const w = wall(ms);
    return `${MON[w.mo - 1]} ${w.d}, ${w.y} ${p2(w.h)}:${p2(w.mi)}`;
  }

  function p2(n) {
    return String(n).padStart(2, "0");
  }

  function fmtInt(n) {
    return nf.format(n);
  }

  // bin edges and durations read at 4 significant digits; a measured value
  // keeps 6, enough for any float32 source without printing its noise
  function fmtNum(v, digits = 4) {
    if (v == null) return "";
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e9)) return Number(v.toPrecision(digits)).toExponential();
    return Number(v.toPrecision(digits)).toLocaleString("en-US", { maximumFractionDigits: 20 });
  }

  function fmtDur(ms) {
    const sign = ms < 0 ? "−" : "";
    const s = Math.round(Math.abs(ms) / 1000);
    if (s < 60) return `${sign}${s} s`;
    const m = Math.round(s / 60);
    if (m < 60) return `${sign}${m} min`;
    if (m < 48 * 60) {
      const h = Math.floor(m / 60);
      return `${sign}${h} h` + (m % 60 ? ` ${m % 60} min` : "");
    }
    const h = Math.round(m / 60);
    return `${sign}${Math.floor(h / 24)} d` + (h % 24 ? ` ${h % 24} h` : "");
  }

  // ---- start ------------------------------------------------------------------

  new ResizeObserver(resize).observe(wrap);
  resize();
  layout();
  setView(ext0, ext1);
  renderPanel();
  document.fonts?.ready.then(() => redraw(ALL));
})();
