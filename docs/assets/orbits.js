// The binary orbit gallery: a tile per binary, a detail view per tile.
// scripts/orbit_gallery.py embeds the data as <script id="og-data"> (same
// shape as docs/assets/orbits.json) next to <div class="og">. All orbit
// curves are sampled in Python; this file only draws them. Every tile is a
// link to #slug, so the browser's back button, deep links and the keyboard
// all work; the prose (guide, badges, notes) comes from the same data.
(function () {
  var REPO = "https://github.com/benjaminpope/";

  function esc(s) { var d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; }
  function num(v) { return typeof v === "number" && isFinite(v); }
  function fmt(v) {
    if (v == null) return "–";
    if (typeof v !== "number") return esc(v);
    if (!isFinite(v)) return "–";
    return (v !== 0 && (Math.abs(v) < 1e-4 || Math.abs(v) >= 1e6)) ? v.toExponential(2) : String(Number(v.toPrecision(4)));
  }
  function f(v) { return Number(v.toFixed(2)); }
  function arr(a) { return Array.isArray(a) ? a : []; }
  function pts(path) { return arr(path).filter(function (p) { return Array.isArray(p) && num(p[0]) && num(p[1]); }); }
  function poly(path, tx, ty, close) {
    var d = "";
    pts(path).forEach(function (p, i) { d += (i ? "L" : "M") + f(tx(p[0])) + " " + f(ty(p[1])); });
    return d && close ? d + "Z" : d;
  }
  function reduced() { return window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches; }

  var PUB_N = 4;
  function pubClass(i) { return "og-p" + (i % PUB_N); }
  var ICON = { agrees: "✓", tension: "!", reported: "◆", awaiting: "…" };
  function statusText(t) {
    return t.status === "awaiting" ? "awaiting " + (t.awaiting || "fit") : t.status;
  }
  var BADGE_TIP = {};
  function badge(t) {
    return '<span class="og-badge og-b-' + esc(t.status) + '"' + (BADGE_TIP[t.status] ? ' title="' + esc(BADGE_TIP[t.status]) + '"' : "") +
      '><span class="og-ico" aria-hidden="true">' + (ICON[t.status] || "") + "</span>" + esc(statusText(t)) + "</span>";
  }

  function flagTags(t) {
    return arr(t.flags).map(function (f) {
      return '<span class="og-flag og-f-' + esc(f.key) + '">' + esc(f.text) + "</span>";
    }).join("");
  }

  // ---- links ----------------------------------------------------------
  function chips(t, compact) {
    var p = t.paper;
    if (!p) return "";
    var h = "";
    function a(href, label, title) {
      return '<a class="og-chip" href="' + esc(href) + '" target="_blank" rel="noopener" title="' + esc(title) + '">' + esc(label) + "</a>";
    }
    if (p.arxiv) h += a("https://arxiv.org/abs/" + encodeURIComponent(p.arxiv), "arXiv:" + p.arxiv, "Open on arXiv");
    if (p.bibcode) h += a("https://ui.adsabs.harvard.edu/abs/" + encodeURIComponent(p.bibcode) + "/abstract", "ADS", "Open on NASA ADS");
    if (!compact && p.doi) h += a("https://doi.org/" + p.doi, "DOI", "Open the journal page");
    return h;
  }

  // ---- geometry helpers -----------------------------------------------
  function niceLen(max) {
    if (!(max > 0)) return 1;
    var e = Math.pow(10, Math.floor(Math.log10(max))), m = max / e;
    return (m >= 5 ? 5 : m >= 2 ? 2 : 1) * e;
  }
  // about n round ticks covering [lo, hi]
  function niceTicks(lo, hi, n) {
    var raw = (hi - lo) / n || 1, e = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / e;
    var step = (m > 5 ? 10 : m > 2.5 ? 5 : m > 2 ? 2.5 : m > 1 ? 2 : 1) * e, out = [];
    for (var v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toPrecision(12)));
    return out;
  }
  // covariance -> polygon of the 1-sigma ellipse in data units
  function ellipse(cov, cx, cy) {
    if (!Array.isArray(cov) || !Array.isArray(cov[0]) || !Array.isArray(cov[1])) return null;
    var a = cov[0][0], b = cov[0][1], d = cov[1][1];
    if (![a, b, d].every(num) || a < 0 || d < 0) return null;
    var tr = (a + d) / 2, det = Math.sqrt(Math.max(0, ((a - d) / 2) * ((a - d) / 2) + b * b));
    var l1 = Math.max(0, tr + det), l2 = Math.max(0, tr - det);
    var th = 0.5 * Math.atan2(2 * b, a - d), out = [];
    for (var k = 0; k < 40; k++) {
      var u = 2 * Math.PI * k / 40, x = Math.sqrt(l1) * Math.cos(u), y = Math.sqrt(l2) * Math.sin(u);
      out.push([cx + x * Math.cos(th) - y * Math.sin(th), cy + x * Math.sin(th) + y * Math.cos(th)]);
    }
    return out;
  }
  function allSkyPoints(t) {
    var out = [];
    if (t.ours) out = out.concat(pts(t.ours.path));
    arr(t.published).forEach(function (a) { out = out.concat(pts(a.path)); });
    arr(t.epochs).forEach(function (e) {
      if (num(e.dra) && num(e.ddec)) out.push([e.dra, e.ddec]);
      if (e.pub && num(e.pub.dra) && num(e.pub.ddec)) out.push([e.pub.dra, e.pub.ddec]);
    });
    return out;
  }
  function bbox(points, pad) {
    var x0 = 0, x1 = 0, y0 = 0, y1 = 0;   // the origin is always in view
    points.forEach(function (p) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); });
    var w = (x1 - x0) || 1, h = (y1 - y0) || 1;
    return { x0: x0 - pad * w, x1: x1 + pad * w, y0: y0 - pad * h, y1: y1 + pad * h };
  }
  function scope(t) {
    if (t.kind === "orbit") return "orbit fit";
    return arr(t.published).length ? "one night fitted, published orbit for context" : "detection positions, no orbit fitted";
  }
  function nights(t) { var n = arr(t.epochs).length; return n ? n + (n === 1 ? " night" : " nights") : ""; }

  // ---- thumbnail ------------------------------------------------------
  function thumb(t) {
    var W = 120, H = 120, m = 9;
    var points = allSkyPoints(t);
    var svg = '<svg class="og-thumb" viewBox="0 0 ' + W + " " + H + '" aria-hidden="true" focusable="false">';
    if (!points.length) {
      return svg + '<circle cx="60" cy="60" r="30" class="og-ph"/><circle cx="60" cy="60" r="3" class="og-star"/><text x="60" y="108" class="og-thumb-txt" text-anchor="middle">' + esc(statusText(t)) + "</text></svg>";
    }
    var b = bbox(points, 0.08);
    var s = Math.min((W - 2 * m) / (b.x1 - b.x0), (H - 2 * m) / (b.y1 - b.y0));
    var ox = (W - s * (b.x1 - b.x0)) / 2, oy = (H - s * (b.y1 - b.y0)) / 2;
    function tx(x) { return ox + (b.x1 - x) * s; }
    function ty(y) { return oy + (b.y1 - y) * s; }
    svg += '<line class="og-grid-l" x1="' + f(tx(0)) + '" x2="' + f(tx(0)) + '" y1="0" y2="' + H + '"/><line class="og-grid-l" y1="' + f(ty(0)) + '" y2="' + f(ty(0)) + '" x1="0" x2="' + W + '"/>';
    var pub = arr(t.published)[0];
    if (pub && pts(pub.path).length) svg += '<path d="' + poly(pub.path, tx, ty, true) + '" class="og-line og-p0 og-dash"/>';
    if (t.ours && pts(t.ours.path).length) svg += '<path d="' + poly(t.ours.path, tx, ty, true) + '" class="og-line og-ours"/>';
    arr(t.epochs).forEach(function (e) {
      if (!num(e.dra) || !num(e.ddec)) return;
      if (e.alias) svg += '<path class="og-x" transform="translate(' + f(tx(e.dra)) + " " + f(ty(e.ddec)) + ')" d="M-3 -3L3 3M-3 3L3 -3"/>';
      else svg += '<circle cx="' + f(tx(e.dra)) + '" cy="' + f(ty(e.ddec)) + '" r="2.8" class="og-dot og-ours"/>';
    });
    return svg + '<circle cx="' + f(tx(0)) + '" cy="' + f(ty(0)) + '" r="3" class="og-star"/></svg>';
  }

  // ---- gallery --------------------------------------------------------
  function gallery(root, data) {
    var h = '<p class="og-lede">Pick a binary. Each thumbnail is the orbit on the sky, east to the left: ' +
      '<span class="og-key og-key-ours"></span>virgil\'s fit, <span class="og-key og-key-pub"></span>the published orbit, ' +
      '<span class="og-key og-key-dot"></span>the nights. The badge says how the numbers compare.</p><div class="og-grid">';
    data.targets.forEach(function (t) {
      var n = nights(t);
      h += '<article class="og-tile og-s-' + esc(t.status) + '">' +
        '<a class="og-open" href="#' + esc(t.slug) + '" data-slug="' + esc(t.slug) + '"><span class="og-thumbwrap">' + thumb(t) + "</span>" +
        '<span class="og-name">' + esc(t.target) + "</span>" +
        '<span class="og-inst">' + esc(t.instrument) + (n ? " · " + n : "") + "</span>" +
        '<span class="og-tag">' + esc(t.tagline) + "</span></a>" + flagTags(t) +
        '<div class="og-foot">' + badge(t) + chips(t, true) + "</div></article>";
    });
    root.innerHTML = h + "</div>";
  }

  // ---- markers --------------------------------------------------------
  // a night: a filled dot, or a cross for a single-night alias peak
  function marker(cx, cy, e, i, key) {
    var lab = "Night " + e.label + (e.alias ? " (single-night alias peak)" : "");
    if (e.alias) {
      return '<g data-key="alias" class="og-x og-hit" tabindex="0" role="img" aria-label="' + esc(lab) + '" data-i="' + i + '" transform="translate(' + f(cx) + " " + f(cy) + ')">' +
        '<circle r="9" class="og-hitarea"/><path d="M-4.5 -4.5L4.5 4.5M-4.5 4.5L4.5 -4.5"/></g>';
    }
    if (e.flag) {
      return '<g data-key="flag" class="og-flag og-hit" tabindex="0" role="img" aria-label="' + esc(lab + " (flagged " + e.flag.code + (e.flag.excluded ? ", excluded from the fit" : ", used in the fit") + ")") + '" data-i="' + i + '" transform="translate(' + f(cx) + " " + f(cy) + ')">' +
        '<circle r="9" class="og-hitarea"/><path d="M0 -5.5L5.5 0L0 5.5L-5.5 0Z"/></g>';
    }
    return '<g data-key="' + key + '" class="og-hit" tabindex="0" role="img" aria-label="' + esc(lab) + '" data-i="' + i + '" transform="translate(' + f(cx) + " " + f(cy) + ')">' +
      '<circle r="9" class="og-hitarea"/><circle r="4.6" class="og-dot og-ours"/></g>';
  }

  // ---- sky plot -------------------------------------------------------
  function skyPlot(t) {
    var W = 480, margin = 12;
    var points = allSkyPoints(t);
    if (!points.length) return { html: '<p class="og-muted">No orbit or astrometry to plot yet.</p>' };
    var b = bbox(points, 0.12);
    var s = Math.min((W - 2 * margin) / (b.x1 - b.x0), 440 / (b.y1 - b.y0));
    var H = Math.max(220, Math.round((b.y1 - b.y0) * s + 2 * margin));
    var ox = (W - s * (b.x1 - b.x0)) / 2, oy = (H - s * (b.y1 - b.y0)) / 2;
    function tx(x) { return ox + (b.x1 - x) * s; }      // RA increases to the left
    function ty(y) { return oy + (b.y1 - y) * s; }      // Dec up
    var g = '<svg class="og-sky" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Sky-plane orbit of ' + esc(t.target) + ': north up, east left">';
    g += '<rect class="og-frame" x="0.5" y="0.5" width="' + (W - 1) + '" height="' + (H - 1) + '" rx="6"/>';
    g += '<line class="og-grid-l" x1="' + f(tx(0)) + '" x2="' + f(tx(0)) + '" y1="0" y2="' + H + '"/>' +
         '<line class="og-grid-l" y1="' + f(ty(0)) + '" y2="' + f(ty(0)) + '" x1="0" x2="' + W + '"/>';
    g += '<clipPath id="og-clip-' + esc(t.slug) + '"><rect x="' + f(tx(b.x1)) + '" y="' + f(ty(b.y1)) + '" width="' + f((b.x1 - b.x0) * s) + '" height="' + f((b.y1 - b.y0) * s) + '"/></clipPath>';
    arr(t.alternatives).forEach(function (a, i) {
      if (pts(a.path).length) g += '<path clip-path="url(#og-clip-' + esc(t.slug) + ')" data-key="alt' + i + '" class="og-line og-alt" d="' + poly(a.path, tx, ty, true) + '"><title>' + esc(a.label) + (num(a.dloss) ? " (Δloss " + fmt(a.dloss) + ")" : "") + "</title></path>";
    });
    arr(t.published).forEach(function (p, i) {
      if (pts(p.path).length) g += '<path data-key="pub' + i + '" class="og-line og-dash ' + pubClass(i) + '" d="' + poly(p.path, tx, ty, true) + '"><title>' + esc(p.label) + "</title></path>";
    });
    if (t.ours && pts(t.ours.path).length) g += '<path data-key="ours" class="og-line og-ours" d="' + poly(t.ours.path, tx, ty, true) + '"><title>' + esc(t.ours.label || "ours") + "</title></path>";
    var hasPub = false, hasAlias = false, hasFlag = false, eps = arr(t.epochs), label = eps.length <= 6;
    eps.forEach(function (e, i) {
      if (!num(e.dra) || !num(e.ddec)) return;
      var el = ellipse(e.cov, e.dra, e.ddec);
      if (e.pub && num(e.pub.dra) && num(e.pub.ddec)) {
        hasPub = true;
        var pe = ellipse(e.pub.cov, e.pub.dra, e.pub.ddec);
        g += '<g data-key="pubep" class="og-ep">' + (pe ? '<path class="og-ell og-p0" d="' + poly(pe, tx, ty, true) + '"/>' : "") +
          '<circle class="og-dot og-hollow og-p0" cx="' + f(tx(e.pub.dra)) + '" cy="' + f(ty(e.pub.ddec)) + '" r="6.4"/></g>';
      }
      if (e.alias) hasAlias = true;
      if (e.flag) hasFlag = true;
      g += '<g data-key="' + (e.alias ? "alias" : "eps") + '" class="og-ep">' + (el ? '<path class="og-ell og-ours' + (e.alias ? " og-ell-alias" : "") + '" d="' + poly(el, tx, ty, true) + '"/>' : "") + "</g>";
    });
    eps.forEach(function (e, i) {
      if (!num(e.dra) || !num(e.ddec)) return;
      g += marker(tx(e.dra), ty(e.ddec), e, i, "eps");
      if (label) g += '<text data-key="' + (e.alias ? "alias" : "eps") + '" class="og-nlab" x="' + f(tx(e.dra) + 9) + '" y="' + f(ty(e.ddec) - 8) + '">' + esc(String(e.label).replace(/^\d{4}-/, "")) + "</text>";
    });
    g += '<circle class="og-star" cx="' + f(tx(0)) + '" cy="' + f(ty(0)) + '" r="6"/>';
    // scale bar and compass
    var L = niceLen(0.22 * (W - 2 * margin) / s), px = L * s, bx = W - margin - px, by = H - margin - 8;
    g += '<g class="og-scale"><line x1="' + f(bx) + '" x2="' + f(bx + px) + '" y1="' + by + '" y2="' + by + '"/><line x1="' + f(bx) + '" x2="' + f(bx) + '" y1="' + (by - 4) + '" y2="' + (by + 4) + '"/><line x1="' + f(bx + px) + '" x2="' + f(bx + px) + '" y1="' + (by - 4) + '" y2="' + (by + 4) + '"/>' +
      '<text x="' + f(bx + px) + '" y="' + (by - 8) + '" text-anchor="end">' + fmt(L) + " mas</text></g>";
    var cx = margin + 28, cy = margin + 32;
    g += '<g class="og-compass"><line x1="' + cx + '" y1="' + cy + '" x2="' + cx + '" y2="' + (cy - 20) + '"/><line x1="' + cx + '" y1="' + cy + '" x2="' + (cx - 20) + '" y2="' + cy + '"/>' +
      '<text x="' + cx + '" y="' + (cy - 25) + '" text-anchor="middle">N</text><text x="' + (cx - 25) + '" y="' + (cy + 5) + '" text-anchor="end">E</text></g>';
    return { html: g + "</svg>", hasPub: hasPub, hasAlias: hasAlias, hasFlag: hasFlag };
  }

  // ---- x/y line plot (phase and date panels) ----------------------------
  // series: [{pts, cls, dash, key}]; dots: [{x, y, e, i, key}] (e = the epoch)
  function linePlot(o) {
    var W = 380, H = 156, l = 42, r = 8, tp = 8, bt = 30;
    var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    function see(x, y) { if (num(x) && num(y)) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); } }
    o.series.forEach(function (s) { s.pts.forEach(function (p) { see(p[0], p[1]); }); });
    arr(o.dots).forEach(function (d) { see(d.x, d.y); });
    if (!isFinite(x0)) return "";
    if (o.xmin != null) x0 = o.xmin;
    if (o.xmax != null) x1 = o.xmax;
    if (x1 === x0) { x0 -= 1; x1 += 1; }
    var yp = ((y1 - y0) || 1) * 0.08; y0 -= yp; y1 += yp;
    if (o.ymin != null) y0 = o.ymin;
    if (o.ymax != null) y1 = o.ymax;
    function X(x) { return l + (x - x0) / (x1 - x0) * (W - l - r); }
    function Y(y) { return tp + (y1 - y) / (y1 - y0) * (H - tp - bt); }
    var g = '<svg class="og-lp" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(o.label) + '">';
    g += '<rect class="og-frame" x="' + l + '" y="' + tp + '" width="' + (W - l - r) + '" height="' + (H - tp - bt) + '"/>';
    (o.yticks || niceTicks(y0, y1, 4)).forEach(function (v) {
      g += '<line class="og-grid-l" x1="' + l + '" x2="' + (W - r) + '" y1="' + f(Y(v)) + '" y2="' + f(Y(v)) + '"/><text class="og-tick" x="' + (l - 6) + '" y="' + f(Y(v) + 4) + '" text-anchor="end">' + fmt(v) + "</text>";
    });
    (o.xticks || niceTicks(x0, x1, 5)).forEach(function (v) {
      g += '<text class="og-tick" x="' + f(X(v)) + '" y="' + (H - bt + 14) + '" text-anchor="middle">' + fmt(v) + "</text>";
    });
    g += '<text class="og-tick og-axl" x="' + (l + (W - l - r) / 2) + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(o.xlabel) + '</text>' +
      '<text class="og-tick og-axl" transform="translate(12 ' + (tp + (H - tp - bt) / 2) + ') rotate(-90)" text-anchor="middle">' + esc(o.ylabel) + "</text>";
    o.series.forEach(function (s) {
      var d = "", prev = null;   // break the line on wraps (PA 360 -> 0) and on decreasing x
      s.pts.forEach(function (p) {
        if (!num(p[0]) || !num(p[1])) { prev = null; return; }
        var brk = !prev || p[0] < prev[0] || (o.wrap && Math.abs(p[1] - prev[1]) > o.wrap);
        d += (brk ? "M" : "L") + f(X(p[0])) + " " + f(Y(p[1]));
        prev = p;
      });
      g += '<path data-key="' + s.key + '" class="og-line ' + s.cls + (s.dash ? " og-dash" : "") + '" d="' + d + '"/>';
    });
    arr(o.dots).forEach(function (d) {
      if (!num(d.x) || !num(d.y)) return;
      if (d.hollow) { g += '<circle data-key="pubep" class="og-dot og-hollow og-p0" cx="' + f(X(d.x)) + '" cy="' + f(Y(d.y)) + '" r="5.6"/>'; return; }
      g += marker(X(d.x), Y(d.y), d.e, d.i, d.key);
    });
    return g + "</svg>";
  }

  function phasePanels(t) {
    var ph = t.phase || {}, ours = arr(ph.ours), pub = arr(ph.published), epochs = arr(t.epochs);
    function series(col) {
      var s = [];
      if (ours.length) s.push({ pts: ours.map(function (p) { return [p[0], p[col]]; }), cls: "og-ours", key: "ours" });
      if (pub.length) s.push({ pts: pub.map(function (p) { return [p[0], p[col]]; }), cls: "og-p0", dash: true, key: "pub0" });
      return s;
    }
    if (ours.length || pub.length) {
      function dots(col) {
        var d = [];
        epochs.forEach(function (e, i) { if (num(e.phase) && num(e[col])) d.push({ x: e.phase, y: e[col], e: e, i: i, key: "eps" }); });
        return d;
      }
      var xt = [0, 0.25, 0.5, 0.75, 1];
      return linePlot({ series: series(1), dots: dots("sep"), xlabel: "orbital phase (0 = periastron)", ylabel: "separation (mas)", xmin: 0, xmax: 1, xticks: xt, ymin: 0, label: "Separation against orbital phase" }) +
        linePlot({ series: series(2), dots: dots("pa"), xlabel: "orbital phase (0 = periastron)", ylabel: "PA (deg)", xmin: 0, xmax: 1, xticks: xt, ymin: 0, ymax: 360, yticks: [0, 90, 180, 270, 360], wrap: 180, label: "Position angle against orbital phase" });
    }
    // no orbit: separation and PA against date, ours (filled) and published (hollow)
    var withSep = epochs.map(function (e, i) { return { e: e, i: i }; }).filter(function (q) { return num(q.e.mjd) && num(q.e.sep); });
    if (!withSep.length) return "";
    function dd(col) {
      var d = withSep.map(function (q) { return { x: q.e.mjd, y: q.e[col], e: q.e, i: q.i, key: "eps" }; });
      withSep.forEach(function (q) { if (q.e.pub && num(q.e.pub[col])) d.push({ x: q.e.mjd, y: q.e.pub[col], hollow: true }); });
      return d;
    }
    return linePlot({ series: [], dots: dd("sep"), xlabel: "MJD", ylabel: "separation (mas)", label: "Separation against MJD" }) +
      linePlot({ series: [], dots: dd("pa"), xlabel: "MJD", ylabel: "PA (deg)", ymin: 0, ymax: 360, yticks: [0, 90, 180, 270, 360], label: "Position angle against MJD" });
  }

  // ---- parameter table ------------------------------------------------
  function pullCell(p) {
    if (!num(p)) return '<td class="og-pull">–</td>';
    var a = Math.abs(p), c = a < 1 ? "og-pl-lo" : a < 3 ? "og-pl-mid" : "og-pl-hi";
    return '<td class="og-pull ' + c + '"><span class="og-bar" aria-hidden="true"><i style="width:' + Math.min(100, a / 5 * 100).toFixed(0) + '%"></i></span><b>' +
      (p < 0 ? "−" : "+") + fmt(Math.round(a * 100) / 100) + "σ</b></td>";
  }
  var STATUS_ICON = { PASS: "✓ ", FAIL: "✕ ", MISSING: "✕ " };
  function table(t) {
    var rows = arr(t.table);
    if (!rows.length) return '<p class="og-muted">No parameter comparison yet' + (t.status === "awaiting" ? ": it appears when the " + (t.awaiting || "fit") + " lands" : "") + ".</p>";
    var scored = rows.filter(function (r) { return r.status === "PASS" || r.status === "FAIL" || r.status === "MISSING"; }).length;
    var h = '<div class="og-tablebox">';
    if (scored && rows.length > 10) h += '<label class="og-tog og-filter"><input type="checkbox" data-filter> Show only the scored rows (' + scored + " of " + rows.length + ")</label>";
    h += '<div class="og-tw"><table class="og-table"><thead><tr><th>Quantity</th><th>virgil</th><th>Published</th><th>Pull</th><th>Status</th></tr></thead><tbody>';
    rows.forEach(function (r) {
      var st = String(r.status || "").toUpperCase();
      var isScored = st === "PASS" || st === "FAIL" || st === "MISSING";
      var cls = st === "PASS" ? "og-ok" : st === "FAIL" || st === "MISSING" ? "og-bad" : "og-dim";
      function pm(v, e) { return fmt(v) + (num(e) ? ' <span class="og-pm">± ' + fmt(e) + "</span>" : ""); }
      h += '<tr class="' + (isScored ? "og-tr-scored" : "og-tr-rep") + '"' + (r.note ? ' title="' + esc(r.note) + '"' : "") + "><td>" + esc(r.quantity) + "</td><td>" + pm(r.ours, r.err) + "</td><td>" + pm(r.pub, r.pub_err) + "</td>" +
        pullCell(r.pull) + '<td class="' + cls + '">' + (STATUS_ICON[st] || "") + esc(r.status) + "</td></tr>";
    });
    return h + "</tbody></table></div></div>";
  }

  // ---- detail ---------------------------------------------------------
  var KIND = { convention: "Convention", caveat: "Caveat", data: "Data" };
  function help(t, data, sky) {
    var has = {
      sky: true, ours: !!(t.ours && pts(t.ours.path).length), pub: arr(t.published).length > 0,
      alt: arr(t.alternatives).length > 0, eps: arr(t.epochs).some(function (e) { return num(e.dra); }),
      alias: !!sky.hasAlias || !!sky.hasFlag, phase: true
    };
    var items = arr(data.guide).filter(function (g) { return has[g.key]; });
    return '<details class="og-help"><summary>How to read this plot</summary><dl>' + items.map(function (g) {
      return "<dt>" + esc(g.title) + "</dt><dd>" + esc(g.text) + "</dd>";
    }).join("") + "</dl></details>";
  }
  function commitLink(repo, sha) {
    var s = String(sha);
    return '<a href="' + REPO + repo + "/commit/" + encodeURIComponent(s) + '" target="_blank" rel="noopener"><code>' + esc(s.slice(0, 7)) + "</code></a>";
  }
  function dataLine(t) {
    var d = t.data || {}, bits = [];
    if (d.source) return "Data: " + esc(d.source) + " products (public, programmes " + esc((d.programmes || []).join(", ")) + "), " + esc(String(d.n_files || 0)) + " OIFITS files, as released by ESO.";
    if (d.archive) return "Data: " + esc(d.archive) + (d.datapi ? "; " + esc(d.datapi) : "") + ". Raw frames are public; we thank the ESO archive and cite the papers.";
    if (d.collection) bits.push("OiDB collection <code>" + esc(d.collection) + "</code>");
    else bits.push("an unpublished (L2) OiDB collection, shown with clearance for display");
    if (d.level && d.collection) bits.push(esc(d.level));
    if (d.datapi) bits.push("dataPI " + esc(d.datapi));
    return "Data: " + bits.join(", ") + (d.collection ? ". We thank the dataPI and cite the paper." : ".");
  }

  function detail(root, t, data, nav) {
    var i = data.targets.indexOf(t), prev = data.targets[(i + data.targets.length - 1) % data.targets.length], next = data.targets[(i + 1) % data.targets.length];
    var h = '<div class="og-detail">' +
      '<div class="og-nav"><a class="og-back" href="#" data-back>← All binaries</a>' +
      '<span class="og-pn"><a href="#' + esc(prev.slug) + '" data-slug="' + esc(prev.slug) + '" title="' + esc(prev.target) + '">‹ prev</a><a href="#' + esc(next.slug) + '" data-slug="' + esc(next.slug) + '" title="' + esc(next.target) + '">next ›</a></span></div>' +
      '<div class="og-dhead"><h2 class="og-dtitle" tabindex="-1">' + esc(t.target) + "</h2>" + badge(t) + '<span class="og-chips">' + chips(t, false) + "</span></div>" +
      '<div class="og-muted og-sub">' + esc(t.instrument) + " · " + esc(scope(t)) + (nights(t) ? " · " + nights(t) : "") + "</div>" +
      '<p class="og-about-t">' + esc(t.about) + "</p>";
    if (t.status === "awaiting") h += '<p class="og-note">The ' + esc(t.awaiting || "fit") + " for this target is not in yet; whatever else exists is shown.</p>";
    else if (t.status_why) h += '<p class="og-why og-why-' + esc(t.status) + '">' + esc(t.status_why) + "</p>";
    h += flagTags(t);
    var sky = skyPlot(t);
    var leg = [];
    if (t.ours && pts(t.ours.path).length) leg.push(["ours", t.ours.label || "ours", "og-ours"]);
    if (arr(t.epochs).some(function (e) { return num(e.dra) && !e.alias; })) leg.push(["eps", "nights (1σ)", "og-ours og-sw-dot"]);
    if (sky.hasAlias) leg.push(["alias", "alias nights (flagged alias-peaks)", "og-ours og-sw-x"]);
    if (sky.hasFlag) {
      var nx = arr(t.epochs).filter(function (e) { return e.flag && e.flag.excluded; }).length, nu = arr(t.epochs).filter(function (e) { return e.flag; }).length - nx;
      leg.push(["flag", "other flagged nights, open diamonds (" + (nx ? nx + " excluded from the fit" : "") + (nx && nu ? ", " : "") + (nu ? nu + " used in the fit" : "") + "; reason in tooltip)", "og-ours og-sw-diamond"]);
    }
    arr(t.published).forEach(function (p, k) { if (pts(p.path).length) leg.push(["pub" + k, p.label || "published", pubClass(k) + " og-sw-dash"]); });
    if (sky.hasPub) leg.push(["pubep", "published nights", "og-p0 og-sw-hollow"]);
    arr(t.alternatives).forEach(function (a, k) {
      if (pts(a.path).length) leg.push(["alt" + k, (a.label || "mode") + (num(a.dloss) ? " (Δloss " + fmt(a.dloss) + ")" : ""), "og-alt"]);
    });
    if (num(t.modes_hidden) && t.modes_hidden > 0) leg.push(["alt-hidden", t.modes_hidden + " alternative mode" + (t.modes_hidden === 1 ? "" : "s") + " with Δloss > " + fmt(t.mode_dloss_max) + " nats not drawn", "og-alt"]);
    var phase = phasePanels(t);
    h += '<div class="og-plots"><figure class="og-skywrap">' + sky.html +
      '<div class="og-legend" role="group" aria-label="Show or hide curves">' + leg.map(function (l) {
        return '<label class="og-tog"><input type="checkbox" checked data-key="' + esc(l[0]) + '"><span class="og-sw ' + l[2] + '"></span>' + esc(l[1]) + "</label>";
      }).join("") + "</div></figure>" + (phase ? '<div class="og-phase">' + phase + "</div>" : "") + "</div>";
    h += help(t, data, sky);
    if (arr(t.notes).length) {
      h += '<h3>Notes</h3><div class="og-notes">' + t.notes.map(function (n) {
        return '<section class="og-notecard og-nk-' + esc(n.kind) + '"><h4><span class="og-kind">' + esc(KIND[n.kind] || n.kind) + "</span> " + esc(n.title) + "</h4><p>" + esc(n.text) + "</p></section>";
      }).join("") + "</div>";
    }
    h += '<h3>Parameters against the paper</h3>' +
      '<p class="og-muted og-pullkey">Pull = (virgil − paper) / the paper\'s error. <span class="og-pl-lo og-pk">below 1σ</span><span class="og-pl-mid og-pk">1 to 3σ</span><span class="og-pl-hi og-pk">above 3σ</span> ' +
      'Bold rows are scored pass or fail; the others are reported only.</p>' + table(t);
    if (arr(t.alternatives).length) {
      h += '<details class="og-alts"><summary>Alternative modes (' + t.alternatives.length + ")</summary><ul>" + t.alternatives.map(function (a) {
        var el = a.elements || {};
        return "<li><strong>" + esc(a.label) + "</strong>" + (num(a.dloss) ? " (Δloss " + fmt(a.dloss) + ")" : "") + ": " +
          Object.keys(el).map(function (k) { return esc(k) + " = " + fmt(el[k]); }).join(", ") + "</li>";
      }).join("") + "</ul></details>";
    }
    h += '<div class="og-refs">';
    if (t.paper) h += '<p class="og-cite"><strong>Reference.</strong> ' + esc(t.paper.cite) + " " + chips(t, false) + "</p>";
    h += '<p class="og-muted">' + dataLine(t) + "</p>";
    var pv = t.provenance || {};
    if (pv.virgil_commit || pv.validation_commit) {
      h += '<p class="og-muted og-prov">Fitted with ' + (pv.virgil_commit ? "virgil " + commitLink("virgil", pv.virgil_commit) : "") +
        (pv.virgil_commit && pv.validation_commit ? " and " : "") + (pv.validation_commit ? "virgil-validation " + commitLink("virgil-validation", pv.validation_commit) : "") + ".</p>";
    }
    h += "</div></div>";
    root.innerHTML = h;
    root.querySelectorAll("[data-back], .og-pn a").forEach(function (a) {
      a.addEventListener("click", function (ev) {
        if (ev.defaultPrevented || ev.button || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
        ev.preventDefault();
        nav(a.getAttribute("data-slug") || "");
      });
    });

    // toggles
    root.querySelectorAll(".og-legend input").forEach(function (cb) {
      cb.addEventListener("change", function () {
        var k = cb.getAttribute("data-key");
        root.querySelectorAll('svg [data-key="' + k + '"]').forEach(function (el) { el.style.display = cb.checked ? "" : "none"; });
      });
    });
    var fl = root.querySelector("[data-filter]");
    if (fl) fl.addEventListener("change", function () { root.querySelector(".og-tablebox").classList.toggle("og-only-scored", fl.checked); });

    // tooltips on the nights; hovering one lights it in every panel
    var tip = document.createElement("div");
    tip.className = "og-tip"; tip.hidden = true;
    root.appendChild(tip);
    function show(ev, e, node) {
      tip.innerHTML = "<strong>" + esc(e.label) + (e.alias ? " · alias peak" : "") + (e.flag ? " · flagged: " + esc(e.flag.code) + (e.flag.excluded ? " (excluded from the fit)" : " (used in the fit)") : "") + "</strong><br>MJD " + fmt(e.mjd) + "<br>sep " + fmt(e.sep) + " mas, PA " + fmt(e.pa) + "°<br>ΔRA " + fmt(e.dra) + ", ΔDec " + fmt(e.ddec) + " mas" +
        (e.pub && num(e.pub.dra) ? "<br>paper: ΔRA " + fmt(e.pub.dra) + ", ΔDec " + fmt(e.pub.ddec) : "") +
        (num(e.v2_scale) ? "<br>V² amplitude × " + fmt(e.v2_scale) : "") + (e.flag ? "<br>" + esc(e.flag.reason) : "") + (e.note ? "<br>" + esc(e.note) : "");
      tip.hidden = false;
      var rr = root.getBoundingClientRect(), tr = node.getBoundingClientRect();
      var x = (ev.clientX != null && ev.type !== "focus" ? ev.clientX : tr.left + tr.width / 2) - rr.left;
      var y = (ev.clientY != null && ev.type !== "focus" ? ev.clientY : tr.top) - rr.top;
      tip.style.left = Math.max(4, Math.min(x + 12, rr.width - tip.offsetWidth - 4)) + "px";
      tip.style.top = (y + 16) + "px";
    }
    function light(i, on) { root.querySelectorAll('.og-hit[data-i="' + i + '"]').forEach(function (n) { n.classList.toggle("og-hl", on); }); }
    root.querySelectorAll(".og-hit").forEach(function (c) {
      var idx = Number(c.getAttribute("data-i")), e = t.epochs[idx];
      ["mouseenter", "mousemove", "focus", "click"].forEach(function (n) { c.addEventListener(n, function (ev) { show(ev, e, c); light(idx, true); }); });
      ["mouseleave", "blur"].forEach(function (n) { c.addEventListener(n, function () { tip.hidden = true; light(idx, false); }); });
    });
  }

  // ---- wiring ---------------------------------------------------------
  function init() {
    var root = document.querySelector(".og");
    if (!root || root.getAttribute("data-og-init")) return;
    root.setAttribute("data-og-init", "1");
    var holder = document.getElementById("og-data"), data = null;
    try { data = holder && JSON.parse(holder.textContent); } catch (err) { data = null; }
    if (!data || !Array.isArray(data.targets) || !data.targets.length) return;   // degrade: leave the static page
    arr(data.badges && data.badges.items).forEach(function (b) { BADGE_TIP[b.key] = b.text; });

    function bySlug(s) { return data.targets.filter(function (t) { return t.slug === s; })[0]; }
    function url(slug) { return location.pathname + location.search + (slug ? "#" + slug : ""); }
    function current() { return bySlug(decodeURIComponent((location.hash || "").slice(1))); }
    var shown = null;   // the slug on screen ("" for the gallery)

    function paint(forFocus) {
      var t = current(), was = shown;
      if (t) {
        detail(root, t, data, nav);
        root.classList.add("og-in-detail");
        shown = t.slug;
        var hd = root.querySelector(".og-dtitle");
        if (hd && forFocus) hd.focus({ preventScroll: true });
        if (root.getBoundingClientRect().top < 0) root.scrollIntoView({ block: "start" });
        return;
      }
      root.classList.remove("og-in-detail");
      gallery(root, data);
      shown = "";
      root.querySelectorAll(".og-open").forEach(function (a) {
        a.addEventListener("click", function (ev) {
          if (ev.defaultPrevented || ev.button || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
          ev.preventDefault();
          nav(a.getAttribute("data-slug"));
        });
      });
      var back = was && root.querySelector('.og-open[data-slug="' + was + '"]');
      if (back && forFocus) back.focus();
    }
    // a thumbnail and the detail's sky plot are one element to the view transition: the tile opens into the plot
    function render(forFocus) {
      var t = current(), want = t ? t.slug : "", painted = root.getAttribute("data-og-painted"), prev = shown;
      if (painted && want === prev) return;
      root.setAttribute("data-og-painted", "1");
      if (!painted || !document.startViewTransition || reduced()) { paint(forFocus); return; }
      var old = prev ? root.querySelector(".og-detail .og-sky") : root.querySelector('.og-open[data-slug="' + want + '"] .og-thumb');
      if (old) old.style.viewTransitionName = "og-hero";
      var vt = document.startViewTransition(function () {
        paint(forFocus);
        var nh = want ? root.querySelector(".og-detail .og-sky") : root.querySelector('.og-open[data-slug="' + prev + '"] .og-thumb');
        if (nh && old) nh.style.viewTransitionName = "og-hero";
      });
      vt.finished.then(function () {
        root.querySelectorAll("[style*=view-transition-name]").forEach(function (n) { n.style.viewTransitionName = ""; });
      }, function () {});
    }
    function nav(slug) {
      if (slug === (shown || "")) return;
      history.pushState(null, "", url(slug));
      render(true);
    }
    root.classList.add("og-ready");
    if (!document.startViewTransition) root.classList.add("og-novt");
    window.__ogRender = function () { render(false); };
    window.__ogNav = nav;
    root.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && shown && !ev.defaultPrevented && !/^(INPUT|SELECT|TEXTAREA)$/.test((ev.target || {}).tagName || "")) nav("");
    });
    render(false);
  }
  function again() { if (window.__ogRender && document.querySelector(".og[data-og-init]")) window.__ogRender(); }
  window.addEventListener("popstate", again);
  window.addEventListener("hashchange", again);
  if (typeof document$ !== "undefined") document$.subscribe(init); else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
