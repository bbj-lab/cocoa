"use strict";
// draws the learned vocabulary from the payload visualizer.py embeds: each binned
// code's training values against its bins, every word, the prefixes, the
// timelines by split, and the tokenizer's config
(function () {
  const D = JSON.parse(document.getElementById("cocoa-data").textContent);
  const app = document.getElementById("app");
  // the page's note for viewers that don't run this script, cleared once drawn
  const fallback = document.getElementById("cocoa-fallback");
  const css = getComputedStyle(document.documentElement);
  const token = (name) => css.getPropertyValue(name).trim();
  const INK2 = token("--ink-2");
  const MUTED = token("--muted");
  const GRID = token("--grid");
  const GRID_SOFT = token("--grid-soft");
  const FONT = css.fontFamily;

  const CARD_CAP = 60; // binned codes drawn at first, and with each "show more"
  const ROW_CAP = 500; // words listed at first, and with each "show more"
  const CARD_H = 64;
  const OPEN_H = 200;
  const CHART_H = 150;
  const STRUCT = new Set(["BOS", "EOS", "UNK", "TIME", "CLCK"]);
  const HOURS = [
    [1 / 60, "1 min"],
    [1 / 6, "10 min"],
    [1, "1 h"],
    [6, "6 h"],
    [24, "1 d"],
    [168, "1 w"],
    [730.5, "1 mo"],
    [8766, "1 y"],
    [87660, "10 y"],
  ];
  const nf = new Intl.NumberFormat("en-US");

  const M = D.meta;
  const C = D.codes;
  const L = D.lanes;
  const P = D.prefixes;
  const B = D.binned;
  const TL = D.timelines; // null without tokens_times.parquet beside the vocabulary
  const V = D.words;
  const nV = V.word.length;

  // ---- what the views look things up by -------------------------------------

  const laneOfPrefix = new Map(P.map((p) => [p.prefix, p.lane]));
  const laneOfWord = (i) => (V.code[i] != null ? C[V.code[i]].lane : laneOfPrefix.get(""));
  const color = (lane) => (lane != null ? L[lane].color : MUTED);
  const laneName = (lane) => (lane != null ? L[lane].name : "");
  const hay = (parts) => parts.filter(Boolean).join("\n").toLowerCase();
  const hayCode = C.map((c) => hay([c.code, c.name, c.category, c.description, laneName(c.lane)]));
  const hayWord = V.word.map((w, i) =>
    hay([w, V.code[i] != null ? hayCode[V.code[i]] : laneName(laneOfWord(i))])
  );
  // a prefix is found by its name, its lane's, or any of its codes'
  const codeHays = new Map();
  C.forEach((c, i) => {
    if (!codeHays.has(c.prefix)) codeHays.set(c.prefix, []);
    codeHays.get(c.prefix).push(hayCode[i]);
  });
  const hayPrefix = P.map((p) =>
    hay([p.prefix, p.category, laneName(p.lane), ...(codeHays.get(p.prefix) || [])])
  );
  const wordsOf = new Map(); // code -> its words
  const binWord = new Map(); // bin -> the unfused word every code shares for it
  for (let i = 0; i < nV; i++) {
    if (V.code[i] != null) {
      if (!wordsOf.has(V.code[i])) wordsOf.set(V.code[i], []);
      wordsOf.get(V.code[i]).push(i);
    } else if (V.kind[i] === "bin") binWord.set(V.bin[i], i);
  }
  const binnedOf = new Map(B.map((b, i) => [b.code, i]));
  const counted = V.count.filter((n) => n != null);
  const trainingTokens = counted.reduce((a, n) => a + n, 0);
  const nCodes = C.filter((c) => !STRUCT.has(c.prefix)).length;
  const nBinsOf = (b) => b.breaks.length + 1;
  // bins that hold training values, or failing a count of them, those that can:
  // cut points that tie leave the bins between them empty
  const used = (b) => (b.bins ? b.bins.filter((n) => n > 0).length : null);
  const reachable = (b) => 1 + new Set(b.breaks.filter((x) => x != null)).size;

  const S = {
    tab: B.length ? "binned" : "words",
    q: "",
    sort: { binned: "code", words: ["token", 1], prefixes: ["order", 1] },
    open: -1, // the binned code whose card is open
    cards: CARD_CAP,
    rows: ROW_CAP,
    group: -1, // the split whose timelines are drawn, or -1 for all of them
  };
  const matches = (hay) => !S.q || hay.includes(S.q);

  // ---- page scaffold ----------------------------------------------------------

  app.append(header());
  const panel = el("section", "panel explorer");
  const tabs = el("div", "tabs");
  tabs.setAttribute("role", "tablist");
  const tabBtns = {};
  for (const [key, label, n] of [
    ["binned", "Binned codes", B.length],
    ["words", "Words", nV],
    ["prefixes", "Prefixes", P.length],
    ...(TL ? [["timelines", "Timelines", TL.all.timelines]] : []),
    ["config", "Tokenizer config", null],
  ]) {
    const b = el("button", "tab", label);
    b.type = "button";
    b.setAttribute("role", "tab");
    if (n != null) b.append(el("span", "ct", fmtInt(n)));
    b.addEventListener("click", () => {
      S.tab = key;
      render();
    });
    tabBtns[key] = b;
    tabs.append(b);
  }
  const bar = el("div", "toolbar");
  const search = el("input");
  search.type = "search";
  search.placeholder = "Find words and codes, e.g. heart_rate";
  search.setAttribute("aria-label", "Find words and codes");
  const glass = el("span", "glass");
  // parsed as html, so the svg needs no namespace url
  glass.innerHTML =
    '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="4.75"/>' +
    '<path d="m10.5 10.5 3.5 3.5"/></svg>';
  const slash = el("kbd", null, "/");
  slash.setAttribute("aria-hidden", "true");
  const field = el("span", "field");
  field.append(glass, search, slash);
  const searchCount = el("span", "count num");
  const searchBox = el("label", "search");
  searchBox.append(field, searchCount);
  const sortSeg = el("div", "seg");
  sortSeg.setAttribute("role", "group");
  sortSeg.setAttribute("aria-label", "Order the binned codes");
  const sortBtns = {};
  for (const [key, label, title] of [
    ["code", "A–Z", "By code"],
    ["values", "Most values", "Most training values first"],
    ["bins", "Fewest bins", "Fewest bins holding values first, as where cut points tie"],
  ]) {
    const b = button(label, title, () => {
      S.sort.binned = key;
      S.cards = CARD_CAP;
      render();
    });
    sortBtns[key] = b;
    sortSeg.append(b);
  }
  const note = el("span", "readout num");
  bar.append(searchBox, sortSeg, note);
  const body = el("div", "vbody");
  body.setAttribute("role", "tabpanel");
  panel.append(tabs, bar, body);
  app.append(panel, footer());

  const tip = el("div", "tip");
  tip.setAttribute("role", "tooltip");
  tip.hidden = true;
  document.body.append(tip);

  // canvases draw once scrolled near, since a vocabulary can bin thousands of codes
  const pending = new Map();
  const io = new IntersectionObserver(
    (entries) => {
      for (const en of entries)
        if (en.isIntersecting && pending.has(en.target)) {
          const draw = pending.get(en.target);
          pending.delete(en.target);
          io.unobserve(en.target);
          draw();
        }
    },
    { rootMargin: "400px 0px" }
  );
  function lazy(canvas, draw) {
    pending.set(canvas, draw);
    io.observe(canvas);
  }

  function render() {
    for (const [k, b] of Object.entries(tabBtns))
      b.setAttribute("aria-selected", String(k === S.tab));
    const searchable = ["binned", "words", "prefixes"].includes(S.tab);
    bar.hidden = !searchable;
    sortSeg.hidden = S.tab !== "binned";
    for (const [k, b] of Object.entries(sortBtns))
      b.setAttribute("aria-pressed", String(k === S.sort.binned));
    io.disconnect();
    pending.clear();
    hideTip();
    body.replaceChildren();
    note.textContent = "";
    searchCount.textContent = "";
    if (S.tab === "binned") renderBinned();
    else if (S.tab === "words") renderWords();
    else if (S.tab === "prefixes") renderPrefixes();
    else if (S.tab === "timelines") renderTimelines();
    else renderConfig();
  }

  function found(n, total, noun) {
    searchCount.textContent = S.q ? `${fmtInt(n)} of ${fmtInt(total)} ${noun}` : "";
  }

  // ---- binned codes -----------------------------------------------------------

  function binnedOrder() {
    const ix = B.map((_, i) => i);
    const n = (i) => B[i].n ?? -1;
    if (S.sort.binned === "values") ix.sort((a, b) => n(b) - n(a) || a - b);
    else if (S.sort.binned === "bins") {
      const k = (i) => used(B[i]) ?? reachable(B[i]);
      ix.sort((a, b) => k(a) - k(b) || n(b) - n(a) || a - b);
    }
    return ix;
  }

  function renderBinned() {
    if (!B.length) {
      body.append(
        el("p", "empty", "No code had a numeric value in training, so the tokenizer binned none.")
      );
      return;
    }
    const list = binnedOrder().filter((i) => matches(hayCode[B[i].code]));
    found(list.length, B.length, "binned codes");
    note.textContent =
      M.value_source === "meds"
        ? M.is_training === false
          ? "Values from this directory's training split, against bins learned on another dataset"
          : "Training values against the bins learned on them"
        : M.bins_source === "vocabulary"
          ? "Each bin's training count, from its words: there are no training values here to draw"
          : "Only cut points: there are no training values here to draw";
    if (!list.length) {
      body.append(el("p", "empty", "No binned code matches."));
      return;
    }
    const grid = el("div", "hists");
    for (const i of list.slice(0, S.cards)) grid.append(binCard(i));
    body.append(grid);
    if (list.length > S.cards)
      body.append(
        more(list.length - S.cards, CARD_CAP, "binned codes", () => {
          S.cards += CARD_CAP;
          render();
        })
      );
    if (S.open >= 0)
      body.querySelector(`.hcard[data-b="${S.open}"]`)?.scrollIntoView({ block: "nearest" });
  }

  function binCard(i) {
    const b = B[i];
    const c = C[b.code];
    const card = el("article", "hcard");
    card.dataset.b = i;
    const hd = el("button", "hcard-hd");
    hd.type = "button";
    hd.setAttribute("aria-expanded", String(S.open === i));
    const sw = el("span", "sw");
    sw.style.background = color(c.lane);
    hd.append(sw, el("span", "hcode", c.code), el("span", "ct", b.n != null ? fmtInt(b.n) : ""));
    hd.title = `${c.code} · click to ${S.open === i ? "close" : "see its bins"}`;
    hd.addEventListener("click", () => toggle(card));
    card.append(hd);
    const sub = [c.description, c.category].filter(Boolean).join(" · ");
    card.append(el("div", "hsub", sub || "\u00a0"));
    const plot = el("div", "hplot");
    const cv = el("canvas");
    cv.setAttribute("aria-hidden", "true");
    // its size until drawn, so the page doesn't shift as cards scroll into view
    cv.style.height = `${b.hist || b.bins ? (S.open === i ? OPEN_H : CARD_H) : 0}px`;
    plot.append(cv);
    plot.addEventListener("click", () => toggle(card));
    hover(cv, (x) => histTip(b, cv.geo, x));
    card.append(plot, el("div", "hax num"), el("div", "hmeta"));
    if (S.open === i) {
      card.classList.add("open");
      card.append(detail(b));
    }
    lazy(cv, () => drawCard(card));
    return card;
  }

  function toggle(card) {
    const i = Number(card.dataset.b);
    const was = S.open;
    if (was >= 0) {
      const old = body.querySelector(`.hcard[data-b="${was}"]`);
      if (old) {
        old.classList.remove("open");
        old.querySelector(".hdetail")?.remove();
        old.querySelector(".hcard-hd").setAttribute("aria-expanded", "false");
        drawCard(old);
      }
    }
    S.open = was === i ? -1 : i;
    if (S.open === i) {
      card.classList.add("open");
      card.append(detail(B[i]));
      card.querySelector(".hcard-hd").setAttribute("aria-expanded", "true");
      drawCard(card);
      card.scrollIntoView({ block: "nearest" });
    }
  }

  function drawCard(card) {
    const b = B[Number(card.dataset.b)];
    const big = card.classList.contains("open");
    const cv = card.querySelector("canvas");
    const w = Math.max(120, Math.floor(cv.parentElement.clientWidth));
    cv.geo = drawHist(cv, b, w, big ? OPEN_H : CARD_H, big);
    const ax = card.querySelector(".hax");
    ax.replaceChildren();
    if (cv.geo && !big) ax.append(...cv.geo.ends.map((s) => el("span", null, s)));
    const meta = card.querySelector(".hmeta");
    meta.replaceChildren(...cardMeta(b));
  }

  function cardMeta(b) {
    const out = [];
    const nb = nBinsOf(b);
    if (b.n === 0) return [el("span", null, "No training values in this directory")];
    const k = used(b);
    const strong = el("b", null, k != null ? `${k} of ${nb}` : `${reachable(b)} of ${nb}`);
    out.push(strong, k != null ? " bins hold values" : " bins can hold values");
    const h = b.hist;
    if (h && (h.under || h.over)) {
      const parts = [];
      if (h.under) parts.push(`${fmtInt(h.under)} below`);
      if (h.over) parts.push(`${fmtInt(h.over)} above`);
      out.push(` · ${parts.join(", ")} the range drawn`);
    }
    if (!h && !b.bins) out.push(" · cut points " + b.breaks.map((x) => fmtNum(x)).join(", "));
    return out;
  }

  // ---- histograms -------------------------------------------------------------

  // where the marks of a binned code's histogram go: its values, by value or in
  // buckets, or failing those, the training count of each bin side by side
  function drawHist(cv, b, w, h, big) {
    const hist = b.hist;
    if (!hist && !b.bins) {
      fit(cv, w, 0);
      return null;
    }
    fit(cv, w, h);
    const ctx = ctx2d(cv);
    ctx.clearRect(0, 0, w, h);
    const pad = big ? { l: 44, r: 14, t: 18, b: 24 } : { l: 2, r: 2, t: 4, b: 1 };
    const gap = big ? 2 : 1;
    const cuts = b.breaks.filter((x) => x != null);
    const g = { pad, w, h, marks: [], cuts: [], regions: [] };
    if (hist && hist.x) {
      // exact values, each a bar of its own at its value
      const xs = hist.x;
      let step = Infinity;
      for (let k = 1; k < xs.length; k++) step = Math.min(step, xs[k] - xs[k - 1]);
      if (!isFinite(step)) step = Math.abs(xs[0]) * 0.2 || 1;
      g.d0 = Math.min(xs[0], ...cuts) - step / 2;
      g.d1 = Math.max(xs[xs.length - 1], ...cuts) + step / 2;
      scale(g);
      const bw = Math.max(1, Math.min(big ? 24 : 12, step * g.k - gap));
      g.marks = xs.map((x, k) => ({ ...around(g.X(x), bw), n: hist.n[k], v: x }));
      g.ends = [fmtNum(xs[0]), fmtNum(xs[xs.length - 1])];
      // a cut point is the lowest value of its bin, so it sits just left of its bar
      const at = (v) => {
        const k = xs.indexOf(v);
        if (k > 0) return (g.X(xs[k - 1]) + g.X(v)) / 2;
        if (k === 0) return g.X(v) - bw / 2 - gap;
        return g.X(v);
      };
      g.cutX = at;
    } else if (hist) {
      // values in buckets of a round width
      const nb = hist.n.length;
      g.d0 = hist.lo;
      g.d1 = hist.lo + hist.w * nb;
      scale(g);
      const px = hist.w * g.k;
      const gp = px > 3 ? gap : 0;
      g.marks = hist.n.map((n, k) => {
        const lo = hist.lo + k * hist.w;
        return { x0: g.X(lo) + gp / 2, x1: g.X(lo + hist.w) - gp / 2, n, lo, hi: lo + hist.w };
      });
      g.ends = [fmtNum(g.d0), fmtNum(g.d1)];
      g.cutX = g.X;
    } else {
      // no values to draw, so each bin's count, side by side
      const nb = b.bins.length;
      g.d0 = 0;
      g.d1 = nb;
      scale(g);
      const slot = g.k;
      const bw = Math.max(1, Math.min(big ? 24 : 16, slot - gap * 2));
      g.marks = b.bins.map((n, k) => ({ ...around(g.X(k + 0.5), bw), n, bin: k }));
      g.ends = ["Q0", `Q${nb - 1}`];
      g.cutX = null;
    }
    const total = b.n || g.marks.reduce((a, m) => a + m.n, 0);
    g.total = total;
    const top = Math.max(1, ...g.marks.map((m) => m.n));
    const ymax = big ? niceCeil(top) : top;
    g.Y = (n) => h - pad.b - (n / ymax) * (h - pad.t - pad.b);
    const base = g.Y(0);

    // the bins as bands, every other one washed, between the cut points
    if (g.cutX) {
      const at = [];
      for (const v of cuts)
        if (!at.length || at[at.length - 1].v !== v) at.push({ v, q: cuts.filter((c) => c <= v).length });
      const edges = [pad.l, ...at.map((c) => g.cutX(c.v)), w - pad.r];
      const qs = [0, ...at.map((c) => c.q)];
      qs.forEach((q, k) => g.regions.push({ x0: edges[k], x1: edges[k + 1], q }));
      g.cuts = at.map((c) => g.cutX(c.v));
      g.regions.forEach((r, k) => {
        if (k % 2) {
          ctx.fillStyle = "rgba(0,0,0,0.035)";
          ctx.fillRect(r.x0, pad.t, r.x1 - r.x0, base - pad.t);
        }
      });
    }
    if (big) {
      ctx.font = `11px ${FONT}`;
      ctx.textBaseline = "middle";
      ctx.textAlign = "right";
      for (const t of countTicks(ymax)) {
        const y = Math.round(g.Y(t));
        ctx.fillStyle = GRID_SOFT;
        ctx.fillRect(pad.l, y, w - pad.l - pad.r, 1);
        ctx.fillStyle = INK2;
        ctx.fillText(fmtCompact(t), pad.l - 6, y);
      }
    }
    const lane = C[b.code].lane;
    ctx.fillStyle = color(lane);
    for (const m of g.marks)
      if (m.n > 0) column(ctx, m.x0, m.x1, Math.min(g.Y(m.n), base - 1), base);
    ctx.fillStyle = GRID;
    ctx.fillRect(pad.l, Math.round(base), w - pad.l - pad.r, 1);
    ctx.fillStyle = INK2;
    for (const x of g.cuts) ctx.fillRect(Math.round(x), pad.t, 1, base - pad.t);
    if (big) {
      // which bin each band is, and the values along the bottom
      ctx.textBaseline = "alphabetic";
      ctx.textAlign = "center";
      for (const r of g.regions) {
        const s = `Q${r.q}`;
        if (r.x1 - r.x0 >= ctx.measureText(s).width + 6)
          ctx.fillText(s, (r.x0 + r.x1) / 2, pad.t - 5);
      }
      if (g.cutX) {
        for (const t of niceTicks(g.d0, g.d1, Math.max(2, Math.floor((w - pad.l - pad.r) / 90)))) {
          const x = g.X(t);
          ctx.fillStyle = GRID;
          ctx.fillRect(Math.round(x), base, 1, 4);
          ctx.fillStyle = INK2;
          fillClamped(ctx, fmtNum(t), x, h - 6, w);
        }
      } else
        for (const m of g.marks) {
          ctx.fillStyle = INK2;
          ctx.fillText(`Q${m.bin}`, (m.x0 + m.x1) / 2, h - 6);
        }
    }
    return g;
  }

  // a mark `w` wide, centered on `x`
  const around = (x, w) => ({ x0: x - w / 2, x1: x + w / 2 });

  function scale(g) {
    const span = g.d1 - g.d0 || 1;
    g.k = (g.w - g.pad.l - g.pad.r) / span;
    g.X = (v) => g.pad.l + (v - g.d0) * g.k;
  }

  // a column from the baseline up, its top corners rounded where it's wide enough
  function column(ctx, x0, x1, top, base) {
    const w = x1 - x0;
    const r = Math.min(2, w / 2, base - top);
    ctx.beginPath();
    if (w >= 4 && r > 0) ctx.roundRect(x0, top, w, base - top, [r, r, 0, 0]);
    else ctx.rect(x0, top, Math.max(1, w), base - top);
    ctx.fill();
  }

  function histTip(b, g, x) {
    if (!g) return null;
    const m = g.marks.find((q) => x >= q.x0 - 3 && x <= q.x1 + 3) || null;
    if (!m) return null;
    const share = g.total ? ` · ${fmtPct(m.n / g.total)}` : "";
    let range;
    if (m.bin != null) range = `Q${m.bin}: ${binRange(b.breaks, m.bin)}`;
    else if (m.v != null) range = `${fmtNum(m.v, 6)}, in Q${binOf(b.breaks, m.v)}`;
    else {
      const q0 = binOf(b.breaks, m.lo);
      const q1 = binOf(b.breaks, m.hi - (m.hi - m.lo) * 1e-6);
      range = `${fmtNum(m.lo)} to ${fmtNum(m.hi)}, in Q${q0}` + (q1 !== q0 ? `–Q${q1}` : "");
    }
    return [`${fmtInt(m.n)}`, `${m.n === 1 ? "value" : "values"}${share}`, range];
  }

  // a value's bin, as the tokenizer bins it: how many cut points are at or below it
  function binOf(breaks, v) {
    let n = 0;
    for (const x of breaks) if (x != null && x <= v) n++;
    return n;
  }

  // whether bin k's cut points tie, so that no value can fall in it
  function tied(br, k) {
    return k > 0 && k < br.length && br[k - 1] != null && br[k] != null && br[k - 1] >= br[k];
  }

  function binRange(br, k) {
    const lo = k > 0 ? br[k - 1] : null;
    const hi = k < br.length ? br[k] : null;
    if (tied(br, k)) return `none, as its cut points tie at ${fmtNum(lo)}`;
    if (lo == null && hi == null) return "every value";
    if (lo == null) return `below ${fmtNum(hi)}`;
    if (hi == null) return `${fmtNum(lo)} and above`;
    return `${fmtNum(lo)} to ${fmtNum(hi)}`;
  }

  // an open card's details: the values' summary and a row per bin
  function detail(b) {
    const c = C[b.code];
    const box = el("div", "hdetail");
    if (b.n) {
      box.append(
        definitions("facts num", [
          ["Training values", fmtInt(b.n)],
          ["Distinct", fmtInt(b.distinct)],
          ["Min", fmtNum(b.min, 6)],
          ["Median", fmtNum(b.median, 6)],
          ["Mean", fmtNum(b.mean, 6)],
          ["Max", fmtNum(b.max, 6)],
        ])
      );
    }
    const total = b.bins ? b.bins.reduce((a, n) => a + n, 0) : 0;
    const most = b.bins ? Math.max(1, ...b.bins) : 1;
    const t = el("table", "events vtable bins");
    const hr = el("tr");
    for (const [name, cls] of [
      ["Bin", null],
      ["Range", "num"],
      [M.bins_source === "vocabulary" ? "Training count" : "Training values", "r num"],
      ["Share", null],
      ["Words", null],
    ])
      hr.append(el("th", cls, name));
    const thead = el("thead");
    thead.append(hr);
    const tb = el("tbody");
    for (let k = 0; k < nBinsOf(b); k++) {
      const tr = el("tr");
      const n = b.bins ? b.bins[k] : null;
      if (n === 0) tr.className = "none";
      tr.append(el("td", "num", `Q${k}`), el("td", "num", binRange(b.breaks, k)));
      tr.append(el("td", "r num", n != null ? fmtInt(n) : "—"));
      const share = el("td", "share");
      if (n != null && total)
        share.append(meter(n / most, color(c.lane)), el("span", "num", fmtPct(n / total)));
      tr.append(share);
      const ws = binWords(b.code, k);
      const td = el("td", "mono");
      if (ws.length) td.textContent = ws.map((i) => `${V.token[i]} ${V.word[i]}`).join("  ");
      else if (!tied(b.breaks, k)) td.append(el("span", "gone", "not in the vocabulary, so UNK"));
      tr.append(td);
      tb.append(tr);
    }
    t.append(thead, tb);
    box.append(t);
    const foot = el("p", "hnote");
    foot.append(
      "A bin holds the values from its lower cut point up to, but not including, its upper one. "
    );
    if (M.bins_source === "vocabulary") foot.append("Counts are of each bin's words in training. ");
    const others = (wordsOf.get(b.code) || []).length;
    const go = el("button", "linkish", others === 1 ? "Show its word" : `Show its ${fmtInt(others)} words`);
    go.type = "button";
    go.addEventListener("click", () => {
      setQuery(c.code);
      S.tab = "words";
      render();
    });
    if (others) foot.append(go);
    box.append(foot);
    return box;
  }

  // the words bin k of a code tokenizes to: fused, its own; unfused, the shared one
  function binWords(code, k) {
    if (!M.fused) return binWord.has(k) ? [binWord.get(k)] : [];
    return (wordsOf.get(code) || []).filter((i) => V.bin[i] === k);
  }

  // ---- words ------------------------------------------------------------------

  function renderWords() {
    const [key, dir] = S.sort.words;
    const list = [];
    for (let i = 0; i < nV; i++) if (matches(hayWord[i])) list.push(i);
    const by = {
      token: (a, b) => V.token[a] - V.token[b],
      word: (a, b) => (V.word[a] < V.word[b] ? -1 : V.word[a] > V.word[b] ? 1 : 0),
      count: (a, b) => (V.count[a] ?? -1) - (V.count[b] ?? -1) || V.token[a] - V.token[b],
    }[key];
    list.sort((a, b) => dir * by(a, b));
    found(list.length, nV, "words");
    if (M.has_counts) {
      const once = counted.filter((n) => n === 1).length;
      const few = counted.filter((n) => n < 10).length;
      note.textContent =
        `${fmtInt(once)} seen once in training, ${fmtInt(few)} fewer than 10 times` +
        (M.min_training_ct > 0 ? `; rarer than ${fmtInt(M.min_training_ct)} became UNK` : "");
    } else note.textContent = "tokenizer.yaml holds no training counts";
    const t = el("table", "events vtable words");
    const cols = [
      ["token", "Token", "r num"],
      ["word", "Word", null],
      [null, "Lane", null],
      [null, "Bin", "r num"],
      [null, "Text", null],
      ["count", "Training count", "r num"],
    ];
    t.append(head(cols, "words", [["count", -1]]));
    const tb = el("tbody");
    for (const i of list.slice(0, S.rows)) {
      const tr = el("tr");
      const bi = V.code[i] != null ? binnedOf.get(V.code[i]) : undefined;
      if (bi != null) {
        tr.className = "go";
        tr.dataset.b = bi;
        tr.title = "Click to see its code's bins";
      }
      const lane = laneOfWord(i);
      const ln = el("td");
      if (lane != null) {
        const k = el("span", "lane-key");
        k.style.background = color(lane);
        ln.append(k, laneName(lane));
      }
      tr.append(
        el("td", "r num", fmtInt(V.token[i])),
        el("td", "mono", V.word[i]),
        ln,
        el("td", "r num", V.bin[i] != null ? `Q${V.bin[i]}` : ""),
        el("td", null, V.text[i] ?? ""),
        el("td", "r num", V.count[i] != null ? fmtInt(V.count[i]) : "")
      );
      tb.append(tr);
    }
    t.append(tb);
    tb.addEventListener("click", (ev) => {
      const tr = ev.target.closest("tr.go");
      if (!tr) return;
      const i = Number(tr.dataset.b);
      setQuery(C[B[i].code].code);
      S.tab = "binned";
      S.open = i;
      render();
    });
    body.append(t);
    if (list.length > S.rows)
      body.append(
        more(list.length - S.rows, ROW_CAP, "words", () => {
          S.rows += ROW_CAP;
          render();
        })
      );
  }

  // a sortable table head: `cols` are [sort key or null, label, class], and
  // `desc` names the keys that sort high to low first
  function head(cols, view, desc) {
    const thead = el("thead");
    const hr = el("tr");
    const [cur, dir] = S.sort[view];
    for (const [key, label, cls] of cols) {
      const th = el("th", cls);
      if (key) {
        const b = el("button", "sort", label);
        b.type = "button";
        if (key === cur) {
          th.setAttribute("aria-sort", dir > 0 ? "ascending" : "descending");
          b.append(el("span", "arrow", dir > 0 ? " ▲" : " ▼"));
        }
        b.addEventListener("click", () => {
          const first = new Map(desc).get(key) || 1;
          S.sort[view] = [key, key === cur ? -dir : first];
          render();
        });
        th.append(b);
      } else th.textContent = label;
      hr.append(th);
    }
    thead.append(hr);
    return thead;
  }

  // ---- prefixes ---------------------------------------------------------------

  function renderPrefixes() {
    const [key, dir] = S.sort.prefixes;
    const list = P.map((_, i) => i).filter((i) => matches(hayPrefix[i]));
    const by = {
      order: (a, b) => a - b,
      words: (a, b) => P[a].words - P[b].words || a - b,
      count: (a, b) => (P[a].count ?? -1) - (P[b].count ?? -1) || a - b,
    }[key];
    list.sort((a, b) => dir * by(a, b));
    found(list.length, P.length, "prefixes");
    note.textContent = M.has_counts ? `${fmtInt(trainingTokens)} tokens in training` : "";
    const most = Math.max(1, ...P.map((p) => p.count || 0));
    const t = el("table", "events vtable prefixes");
    t.append(
      head(
        [
          ["order", "Prefix", null],
          [null, "Name", null],
          [null, "Lane", null],
          ["words", "Words", "r num"],
          [null, "Codes", "r num"],
          [null, "Binned", "r num"],
          ["count", "Training tokens", null],
        ],
        "prefixes",
        [
          ["words", -1],
          ["count", -1],
        ]
      )
    );
    const tb = el("tbody");
    for (const i of list) {
      const p = P[i];
      const tr = el("tr");
      if (p.prefix) {
        tr.className = "go";
        // a word of its own, as BOS is, or a prefix to its codes
        tr.dataset.q = STRUCT.has(p.prefix) && p.codes <= 1 ? p.prefix : `${p.prefix}//`;
        tr.title = `Click to list its words`;
      }
      const ln = el("td");
      if (p.lane != null) {
        const k = el("span", "lane-key");
        k.style.background = color(p.lane);
        ln.append(k, laneName(p.lane));
      }
      const share = el("td", "share");
      if (p.count != null) {
        share.append(meter(p.count / most, color(p.lane)), el("span", "num", fmtInt(p.count)));
        if (trainingTokens) share.append(el("span", "pct num", fmtPct(p.count / trainingTokens)));
      }
      tr.append(
        el("td", "mono", p.prefix || "—"),
        el("td", null, p.prefix ? p.category || "" : "Unfused bins and text values, shared by codes"),
        ln,
        el("td", "r num", fmtInt(p.words)),
        el("td", "r num", fmtInt(p.codes)),
        el("td", "r num", p.binned ? fmtInt(p.binned) : ""),
        share
      );
      tb.append(tr);
    }
    tb.addEventListener("click", (ev) => {
      const tr = ev.target.closest("tr.go");
      if (!tr) return;
      setQuery(tr.dataset.q);
      S.tab = "words";
      render();
    });
    t.append(tb);
    body.append(t);
  }

  // ---- timelines --------------------------------------------------------------

  function renderTimelines() {
    const groups = [TL.all, ...TL.splits];
    const name = (g) => (g === TL.all ? "All" : g.split ?? "No split");
    const t = el("table", "events vtable splits");
    const hr = el("tr");
    for (const [label, cls] of [
      ["Split", null],
      ["Timelines", "r num"],
      ["Tokens", "r num"],
      ["Out of vocabulary", "r num"],
      ["Mean length", "r num"],
      ["Median length", "r num"],
      ["Mean duration", "r num"],
      ["Median duration", "r num"],
      ["First event", "num"],
      ["Last event", "num"],
    ])
      hr.append(el("th", cls, label));
    const thead = el("thead");
    thead.append(hr);
    const tb = el("tbody");
    for (const g of [...TL.splits, TL.all]) {
      const tr = el("tr", g === TL.all ? "total" : null);
      tr.append(
        el("td", null, name(g)),
        el("td", "r num", fmtInt(g.timelines)),
        el("td", "r num", fmtInt(g.tokens)),
        el("td", "r num", `${fmtInt(g.unk)}` + (g.tokens ? ` (${fmtPct(g.unk / g.tokens)})` : "")),
        el("td", "r num", fmtNum(g.length.mean)),
        el("td", "r num", fmtNum(g.length.median)),
        el("td", "r num", g.hours.mean != null ? fmtDur(g.hours.mean * 36e5) : ""),
        el("td", "r num", g.hours.median != null ? fmtDur(g.hours.median * 36e5) : ""),
        el("td", "num", g.first ? g.first.label : ""),
        el("td", "num", g.last ? g.last.label : "")
      );
      tb.append(tr);
    }
    t.append(thead, tb);
    const scroll = el("div", "hscroll");
    scroll.append(t);
    body.append(scroll);
    body.append(
      el(
        "p",
        "hnote",
        "As tokenize --verbose summarizes them, from tokens_times.parquet" +
          (TL.timezone ? `; times in ${TL.timezone}.` : ".")
      )
    );

    const seg = el("div", "seg");
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "Whose timelines to draw");
    groups.forEach((g, k) => {
      const title = g === TL.all ? "Draw every timeline" : `Draw the timelines of ${name(g)}`;
      const b = button(name(g), title, () => {
        S.group = k - 1;
        render();
      });
      b.setAttribute("aria-pressed", String(S.group === k - 1));
      seg.append(b);
    });
    const pick = el("div", "pick");
    pick.append(seg);
    const g = groups[S.group + 1];
    if (g.instant)
      pick.append(el("span", "count", `${fmtInt(g.instant)} span no time, so have no duration to draw`));
    body.append(pick);
    const charts = el("div", "charts");
    charts.append(
      logChart("Tokens per timeline", g.lengths, TL.length_from, g.timelines, "length", g.length),
      logChart(
        "Duration, from first token to last",
        g.durations,
        TL.hours_from,
        g.timelines - g.instant,
        "hours",
        g.hours
      )
    );
    body.append(charts);
  }

  function logChart(title, counts, from, total, kind, stats) {
    const fig = el("figure", "chart");
    fig.append(el("figcaption", null, title));
    const cv = el("canvas");
    cv.setAttribute("aria-hidden", "true");
    cv.style.height = `${CHART_H}px`;
    const plot = el("div", "cplot");
    plot.append(cv);
    fig.append(plot);
    const fmt = kind === "hours" ? (h) => fmtDur(h * 36e5) : (n) => fmtNum(n);
    if (stats && stats.min != null) {
      const line =
        `min ${fmt(stats.min)} · quartiles ${fmt(stats.q25)}, ${fmt(stats.median)}, ` +
        `${fmt(stats.q75)} · max ${fmt(stats.max)} · mean ${fmt(stats.mean)}` +
        (stats.std != null ? ` (sd ${fmt(stats.std)})` : "");
      fig.append(el("div", "cstats num", line));
    }
    hover(cv, (x) => {
      const g = cv.geo;
      if (!g) return null;
      const k = Math.floor((x - g.pad.l) / g.slot);
      if (k < 0 || k >= counts.length) return null;
      const lo = 10 ** ((from + k) / TL.per_decade);
      const hi = 10 ** ((from + k + 1) / TL.per_decade);
      const n = counts[k];
      // a length is a whole number of tokens, at least lo and less than hi
      const [a, z] = [Math.ceil(lo - 1e-9), Math.ceil(hi - 1e-9) - 1];
      const range =
        kind === "hours"
          ? `${fmtDur(lo * 36e5)} to ${fmtDur(hi * 36e5)}`
          : a < z
            ? `${fmtInt(a)} to ${fmtInt(z)} tokens`
            : `${fmtInt(a)} tokens`;
      const share = total ? ` · ${fmtPct(n / total)}` : "";
      return [fmtInt(n), `${n === 1 ? "timeline" : "timelines"}${share}`, range];
    });
    lazy(cv, () => {
      cv.geo = drawLog(cv, counts, from, kind);
    });
    return fig;
  }

  // a histogram on a log axis, PER_DECADE buckets to a power of ten
  function drawLog(cv, counts, from, kind) {
    const w = Math.max(200, Math.floor(cv.parentElement.clientWidth));
    const h = CHART_H;
    fit(cv, w, h);
    const ctx = ctx2d(cv);
    ctx.clearRect(0, 0, w, h);
    const pad = { l: 44, r: 12, t: 8, b: 24 };
    const n = counts.length;
    if (!n) return null;
    const slot = (w - pad.l - pad.r) / n;
    const top = niceCeil(Math.max(1, ...counts));
    const Y = (c) => h - pad.b - (c / top) * (h - pad.t - pad.b);
    const base = Y(0);
    ctx.font = `11px ${FONT}`;
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (const t of countTicks(top)) {
      const y = Math.round(Y(t));
      ctx.fillStyle = GRID_SOFT;
      ctx.fillRect(pad.l, y, w - pad.l - pad.r, 1);
      ctx.fillStyle = INK2;
      ctx.fillText(fmtCompact(t), pad.l - 6, y);
    }
    ctx.fillStyle = INK2;
    const gap = slot > 4 ? 2 : 0;
    counts.forEach((c, k) => {
      const x0 = pad.l + k * slot;
      if (c > 0) column(ctx, x0 + gap / 2, x0 + slot - gap / 2, Math.min(Y(c), base - 1), base);
    });
    ctx.fillStyle = GRID;
    ctx.fillRect(pad.l, Math.round(base), w - pad.l - pad.r, 1);
    // ticks at round values: powers of ten of tokens, familiar spans of time
    const X = (v) => pad.l + (Math.log10(v) * TL.per_decade - from) * slot;
    const ticks = [];
    const lo = 10 ** (from / TL.per_decade);
    const hi = 10 ** ((from + n) / TL.per_decade);
    if (kind === "hours") for (const [v, s] of HOURS) if (v >= lo && v <= hi) ticks.push([v, s]);
    if (kind !== "hours" || ticks.length < 2) {
      ticks.length = 0;
      for (let d = Math.ceil(Math.log10(lo)); 10 ** d <= hi; d++)
        ticks.push([10 ** d, kind === "hours" ? fmtDur(10 ** d * 36e5) : fmtCompact(10 ** d)]);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    for (const [v, s] of ticks) {
      const x = X(v);
      ctx.fillStyle = GRID;
      ctx.fillRect(Math.round(x), base, 1, 4);
      ctx.fillStyle = INK2;
      fillClamped(ctx, s, x, h - 6, w);
    }
    return { pad, slot };
  }

  // ---- tokenizer config -------------------------------------------------------

  function renderConfig() {
    const facts = [];
    if (M.is_training != null)
      facts.push([
        "Learned",
        M.is_training ? "on this directory's training split" : "on another dataset, applied here",
      ]);
    if (M.tokenizer_created) facts.push(["Created", M.tokenizer_created]);
    if (M.tokenizer_version) facts.push(["By cocoa", M.tokenizer_version]);
    body.append(definitions("facts", facts));
    const dl = el("dl", "config");
    for (const [k, v] of D.config) dl.append(el("dt", null, k), el("dd", null, v));
    body.append(el("h2", "sub", "As saved in tokenizer.yaml"), dl);
  }

  // ---- header, footer ---------------------------------------------------------

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
    title.append(el("h1", null, "Vocabulary"));
    const fused = el("span", "badge", M.fused ? "fused" : "unfused");
    fused.title = M.fused
      ? "Each event is one word: its code, bin, and text value together"
      : "An event's code, bin, and text value are words of their own";
    title.append(fused);
    if (M.is_training === false) {
      const b = el("span", "badge", "transferred");
      b.title = "Learned on another dataset, and applied here with tokenize --tokenizer-home";
      title.append(b);
    }
    hd.append(title);
    const facts = [
      ["Words", fmtInt(M.vocab_size)],
      ["Codes", fmtInt(nCodes)],
      ["Binned codes", fmtInt(B.length)],
    ];
    if (M.n_bins != null) facts.push(["Bins per code", fmtInt(M.n_bins)]);
    if (M.has_counts) facts.push(["Tokens in training", fmtInt(trainingTokens)]);
    if (TL) {
      facts.push(["Timelines", fmtInt(TL.all.timelines)]);
      if (TL.all.length.median != null) facts.push(["Median length", fmtNum(TL.all.length.median)]);
    }
    hd.append(definitions("facts num", facts));
    if (M.is_training === false)
      hd.append(
        el(
          "p",
          "lede",
          "This vocabulary was learned on another dataset, so its words and training counts are " +
            "that dataset's, while the histograms are of this directory's training split."
        )
      );
    return hd;
  }

  function footer() {
    const f = el("footer", "meta");
    for (const s of [
      `cocoa ${M.cocoa_version}`,
      `rendered ${M.generated}`,
      M.tokenizer_created ? `tokenizer created ${M.tokenizer_created}` : null,
      M.tokenizer_version ? `by cocoa ${M.tokenizer_version}` : null,
      M.processed_data_home,
    ])
      if (s) f.append(el("span", null, s));
    return f;
  }

  // ---- tooltip ----------------------------------------------------------------

  // `at(x)` gives the [value, label, detail] under x in the canvas, or null
  function hover(cv, at) {
    cv.addEventListener("pointermove", (ev) => {
      const r = cv.getBoundingClientRect();
      const lines = at(ev.clientX - r.left);
      if (!lines) return hideTip();
      tip.replaceChildren();
      const line = el("div", "tip-head");
      line.append(el("strong", "num", lines[0]), el("span", null, lines[1]));
      tip.append(line, el("div", "tip-meta num", lines[2]));
      tip.hidden = false;
      placeTip(ev.clientX, ev.clientY);
    });
    cv.addEventListener("pointerleave", hideTip);
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

  // ---- search -----------------------------------------------------------------

  function setQuery(q) {
    search.value = q;
    S.q = q.trim().toLowerCase();
    S.cards = CARD_CAP;
    S.rows = ROW_CAP;
  }

  let searchTimer = 0;
  search.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      setQuery(search.value);
      render();
    }, 120);
  });
  search.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      setQuery("");
      render();
    }
  });
  document.addEventListener("keydown", (ev) => {
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName)) return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (ev.key === "/" && !bar.hidden) {
      search.focus();
      search.select();
    } else if (ev.key === "Escape" && S.open >= 0) {
      const card = body.querySelector(`.hcard[data-b="${S.open}"]`);
      if (card) toggle(card);
      else S.open = -1;
    } else return;
    ev.preventDefault();
  });

  // ---- helpers ----------------------------------------------------------------

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function button(label, title, fn) {
    const b = el("button", "btn", label);
    b.type = "button";
    b.title = title;
    b.addEventListener("click", fn);
    return b;
  }

  // a button to show `step` more of the `n` matches not yet shown
  function more(n, step, noun, fn) {
    const p = el("p", "more");
    const k = Math.min(n, step);
    p.append(button(`Show ${fmtInt(k)} more`, `Show ${fmtInt(k)} more ${noun}`, fn));
    p.append(el("span", "count", `${fmtInt(n)} ${noun} not shown yet`));
    return p;
  }

  function meter(frac, c) {
    const m = el("span", "meter");
    const f = el("span", "fill");
    f.style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
    f.style.background = c;
    m.append(f);
    return m;
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

  function fillClamped(ctx, s, x, y, w) {
    const tw = ctx.measureText(s).width;
    ctx.fillText(s, Math.max(tw / 2 + 2, Math.min(w - tw / 2 - 2, x)), y);
  }

  // round ticks across [a, b], about n of them
  function niceTicks(a, b, n) {
    const raw = (b - a) / Math.max(1, n);
    const p = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw);
    const out = [];
    for (let t = Math.ceil(a / step) * step; t <= b + step * 1e-9; t += step)
      out.push(Number(t.toPrecision(12)));
    return out;
  }

  // ticks up a count axis to `top`, halfway too where that's a whole count
  function countTicks(top) {
    return top % 2 ? [0, top] : [0, top / 2, top];
  }

  function niceCeil(v) {
    const p = 10 ** Math.floor(Math.log10(v));
    return [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= v);
  }

  function fmtInt(n) {
    return nf.format(n);
  }

  function fmtCompact(n) {
    if (n >= 1e6) return `${Number((n / 1e6).toPrecision(3))}M`;
    if (n >= 1e3) return `${Number((n / 1e3).toPrecision(3))}k`;
    return fmtNum(n);
  }

  function fmtPct(f) {
    const p = f * 100;
    if (p === 0) return "0%";
    if (p < 0.1) return "<0.1%";
    return `${p < 10 ? p.toFixed(1) : Math.round(p)}%`;
  }

  // bin edges and summaries read at 4 significant digits; a measured value
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

  let resizeTimer = 0;
  let lastW = innerWidth;
  addEventListener("resize", () => {
    if (innerWidth === lastW) return;
    lastW = innerWidth;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(render, 150);
  });
  render();
  fallback?.remove();
  document.fonts?.ready.then(render);
})();
