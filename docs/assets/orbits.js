// The binary orbit gallery: a tile per binary, a detail view per tile.
// scripts/orbit_gallery.py embeds the data as <script id="og-data"> (same
// shape as docs/assets/orbits.json) next to <div class="og">. All orbit
// curves are sampled in Python; this file only draws them.
(function () {
  var NS = "http://www.w3.org/2000/svg";

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
  var PUB_N = 4;
  function pubClass(i) { return "og-p" + (i % PUB_N); }
  var STATUS = { agrees: "agrees", tension: "tension", reported: "reported", awaiting: "awaiting fit" };
  function statusText(s) { return STATUS[s] || s || ""; }
  function arxivUrl(id) { return "https://arxiv.org/abs/" + encodeURIComponent(id); }
  function chip(t) {
    var id = t.paper && t.paper.arxiv;
    return id ? '<a class="og-chip" href="' + esc(arxivUrl(id)) + '" target="_blank" rel="noopener" title="Open on arXiv">arXiv:' + esc(id) + "</a>" : "";
  }

  // A nice length (1, 2, 5 x 10^n) not above `max`.
  function niceLen(max) {
    if (!(max > 0)) return 1;
    var e = Math.pow(10, Math.floor(Math.log10(max))), m = max / e;
    return (m >= 5 ? 5 : m >= 2 ? 2 : 1) * e;
  }

  // Covariance -> polygon of the 1-sigma ellipse in data units.
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
    arr(t.alternatives).forEach(function (a) { out = out.concat(pts(a.path)); });
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

  // ---- thumbnail ------------------------------------------------------
  function thumb(t) {
    var W = 120, H = 120, m = 8;
    var points = allSkyPoints(t);
    var svg = '<svg class="og-thumb" viewBox="0 0 ' + W + " " + H + '" aria-hidden="true" focusable="false">';
    if (!points.length) {
      return svg + '<text x="60" y="64" class="og-thumb-txt" text-anchor="middle">awaiting fit</text></svg>';
    }
    var b = bbox(points, 0.06);
    var s = Math.min((W - 2 * m) / (b.x1 - b.x0), (H - 2 * m) / (b.y1 - b.y0));
    var ox = (W - s * (b.x1 - b.x0)) / 2, oy = (H - s * (b.y1 - b.y0)) / 2;
    function tx(x) { return ox + (b.x1 - x) * s; }
    function ty(y) { return oy + (b.y1 - y) * s; }
    svg += '<circle cx="' + f(tx(0)) + '" cy="' + f(ty(0)) + '" r="2.6" class="og-star"/>';
    var pub = arr(t.published)[0];
    if (pub && pts(pub.path).length) svg += '<path d="' + poly(pub.path, tx, ty, true) + '" class="og-line og-p0 og-dash"/>';
    if (t.ours && pts(t.ours.path).length) svg += '<path d="' + poly(t.ours.path, tx, ty, true) + '" class="og-line og-ours"/>';
    arr(t.epochs).forEach(function (e) {
      if (num(e.dra) && num(e.ddec)) svg += '<circle cx="' + f(tx(e.dra)) + '" cy="' + f(ty(e.ddec)) + '" r="3.4" class="og-dot og-ours"/>';
    });
    return svg + "</svg>";
  }

  // ---- gallery --------------------------------------------------------
  function gallery(root, data, open) {
    var h = '<div class="og-grid">';
    data.targets.forEach(function (t) {
      var n = arr(t.epochs).length;
      h += '<article class="og-tile og-s-' + esc(t.status) + '">' +
        '<button type="button" class="og-open" data-slug="' + esc(t.slug) + '" aria-label="Open ' + esc(t.target) + '">' + thumb(t) +
        '<span class="og-name">' + esc(t.target) + "</span>" +
        '<span class="og-inst">' + esc(t.instrument) + (n ? " · " + n + " epoch" + (n === 1 ? "" : "s") : "") + "</span></button>" +
        '<div class="og-foot"><span class="og-badge og-b-' + esc(t.status) + '">' + esc(statusText(t.status)) + "</span>" +
        (t.kind === "positions" ? '<span class="og-badge og-b-kind">positions only</span>' : "") + chip(t) + "</div></article>";
    });
    root.innerHTML = h + "</div>";
    root.querySelectorAll(".og-open").forEach(function (b) {
      b.addEventListener("click", function () { open(b.getAttribute("data-slug")); });
    });
  }

  // ---- sky plot -------------------------------------------------------
  function skyPlot(t, tip) {
    var W = 640, margin = 14;
    var points = allSkyPoints(t);
    if (!points.length) return { html: '<p class="og-muted">No orbit or astrometry to plot yet.</p>', keys: [] };
    var b = bbox(points, 0.12);
    var s = Math.min((W - 2 * margin) / (b.x1 - b.x0), 560 / (b.y1 - b.y0));
    var H = Math.max(260, Math.round((b.y1 - b.y0) * s + 2 * margin));
    var ox = (W - s * (b.x1 - b.x0)) / 2, oy = (H - s * (b.y1 - b.y0)) / 2;
    function tx(x) { return ox + (b.x1 - x) * s; }      // RA increases to the left
    function ty(y) { return oy + (b.y1 - y) * s; }      // Dec up
    var g = '<svg class="og-sky" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Sky-plane orbit of ' + esc(t.target) + '">';
    g += '<rect class="og-frame" x="0.5" y="0.5" width="' + (W - 1) + '" height="' + (H - 1) + '"/>';
    g += '<line class="og-grid-l" x1="' + f(tx(0)) + '" x2="' + f(tx(0)) + '" y1="0" y2="' + H + '"/>' +
         '<line class="og-grid-l" y1="' + f(ty(0)) + '" y2="' + f(ty(0)) + '" x1="0" x2="' + W + '"/>';
    // alternatives, then published, then ours on top
    arr(t.alternatives).forEach(function (a, i) {
      if (pts(a.path).length) g += '<path data-key="alt' + i + '" class="og-line og-alt" d="' + poly(a.path, tx, ty, true) + '"><title>' + esc(a.label) + (num(a.dloss) ? " (Δloss " + fmt(a.dloss) + ")" : "") + "</title></path>";
    });
    arr(t.published).forEach(function (p, i) {
      if (pts(p.path).length) g += '<path data-key="pub' + i + '" class="og-line og-dash ' + pubClass(i) + '" d="' + poly(p.path, tx, ty, true) + '"><title>' + esc(p.label) + "</title></path>";
    });
    if (t.ours && pts(t.ours.path).length) g += '<path data-key="ours" class="og-line og-ours" d="' + poly(t.ours.path, tx, ty, true) + '"><title>' + esc(t.ours.label || "ours") + "</title></path>";
    // epochs
    var hasPub = false;
    arr(t.epochs).forEach(function (e, i) {
      if (!num(e.dra) || !num(e.ddec)) return;
      var el = ellipse(e.cov, e.dra, e.ddec);
      if (e.pub && num(e.pub.dra) && num(e.pub.ddec)) {
        hasPub = true;
        var pe = ellipse(e.pub.cov, e.pub.dra, e.pub.ddec);
        g += '<g data-key="pubep" class="og-ep og-pubep">' + (pe ? '<path class="og-ell og-p0" d="' + poly(pe, tx, ty, true) + '"/>' : "") +
          '<circle class="og-dot og-hollow og-p0" cx="' + f(tx(e.pub.dra)) + '" cy="' + f(ty(e.pub.ddec)) + '" r="5.2"/></g>';
      }
      g += '<g data-key="eps" class="og-ep">' + (el ? '<path class="og-ell og-ours" d="' + poly(el, tx, ty, true) + '"/>' : "") +
        '<circle class="og-dot og-ours og-hit" tabindex="0" data-i="' + i + '" cx="' + f(tx(e.dra)) + '" cy="' + f(ty(e.ddec)) + '" r="6.2"/></g>';
    });
    // primary
    g += '<circle class="og-star" cx="' + f(tx(0)) + '" cy="' + f(ty(0)) + '" r="5"/>';
    // scale bar
    var L = niceLen(0.22 * (W - 2 * margin) / s), px = L * s, bx = W - margin - px, by = H - margin - 8;
    g += '<g class="og-scale"><line x1="' + f(bx) + '" x2="' + f(bx + px) + '" y1="' + by + '" y2="' + by + '"/><line x1="' + f(bx) + '" x2="' + f(bx) + '" y1="' + (by - 4) + '" y2="' + (by + 4) + '"/><line x1="' + f(bx + px) + '" x2="' + f(bx + px) + '" y1="' + (by - 4) + '" y2="' + (by + 4) + '"/>' +
      '<text x="' + f(bx + px / 2) + '" y="' + (by - 8) + '" text-anchor="middle">' + fmt(L) + " mas</text></g>";
    // compass: E left, N up
    var cx = margin + 22, cy = margin + 26;
    g += '<g class="og-compass"><line x1="' + cx + '" y1="' + cy + '" x2="' + cx + '" y2="' + (cy - 18) + '"/><line x1="' + cx + '" y1="' + cy + '" x2="' + (cx - 18) + '" y2="' + cy + '"/>' +
      '<text x="' + cx + '" y="' + (cy - 22) + '" text-anchor="middle">N</text><text x="' + (cx - 22) + '" y="' + (cy + 4) + '" text-anchor="end">E</text></g>';
    g += "</svg>";
    return { html: g, hasPub: hasPub };
  }

  // ---- generic x/y line plot (phase panels) ---------------------------
  // series: [{pts:[[x,y],...], cls, dash, key}], dots: [{x,y,cls,key}]
  function linePlot(o) {
    var W = 640, H = 190, l = 46, r = 12, tp = 10, bt = 30;
    var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    function see(x, y) { if (num(x) && num(y)) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); } }
    o.series.forEach(function (s) { s.pts.forEach(function (p) { see(p[0], p[1]); }); });
    arr(o.dots).forEach(function (d) { see(d.x, d.y); });
    if (!isFinite(x0)) return "";
    if (o.xmin != null) x0 = o.xmin;
    if (o.xmax != null) x1 = o.xmax;
    if (x1 === x0) { x0 -= 1; x1 += 1; }
    var yp = ((y1 - y0) || 1) * 0.08; y0 -= yp; y1 += yp;
    if (o.ymin != null) y0 = Math.max(o.ymin, y0);
    if (o.ymax != null) y1 = Math.min(o.ymax, y1);
    function X(x) { return l + (x - x0) / (x1 - x0) * (W - l - r); }
    function Y(y) { return tp + (y1 - y) / (y1 - y0) * (H - tp - bt); }
    var g = '<svg class="og-lp" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(o.label) + '">';
    g += '<rect class="og-frame" x="' + l + '" y="' + tp + '" width="' + (W - l - r) + '" height="' + (H - tp - bt) + '"/>';
    var ny = 4, nx = 5, i, v;
    for (i = 0; i <= ny; i++) { v = y0 + (y1 - y0) * i / ny; g += '<line class="og-grid-l" x1="' + l + '" x2="' + (W - r) + '" y1="' + f(Y(v)) + '" y2="' + f(Y(v)) + '"/><text class="og-tick" x="' + (l - 5) + '" y="' + f(Y(v) + 4) + '" text-anchor="end">' + fmt(Number(v.toPrecision(3))) + "</text>"; }
    for (i = 0; i <= nx; i++) { v = x0 + (x1 - x0) * i / nx; g += '<text class="og-tick" x="' + f(X(v)) + '" y="' + (H - bt + 14) + '" text-anchor="middle">' + fmt(Number(v.toPrecision(o.xprec || 3))) + "</text>"; }
    g += '<text class="og-tick og-axl" x="' + (l + (W - l - r) / 2) + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(o.xlabel) + '</text>' +
      '<text class="og-tick og-axl" transform="translate(11 ' + (tp + (H - tp - bt) / 2) + ') rotate(-90)" text-anchor="middle">' + esc(o.ylabel) + "</text>";
    o.series.forEach(function (s) {
      // break the line on wraps (PA 360 -> 0) and on decreasing x
      var d = "", prev = null;
      s.pts.forEach(function (p) {
        if (!num(p[0]) || !num(p[1])) { prev = null; return; }
        var brk = !prev || p[0] < prev[0] || (o.wrap && Math.abs(p[1] - prev[1]) > o.wrap);
        d += (brk ? "M" : "L") + f(X(p[0])) + " " + f(Y(p[1]));
        prev = p;
      });
      g += '<path data-key="' + s.key + '" class="og-line ' + s.cls + (s.dash ? " og-dash" : "") + '" d="' + d + '"/>';
    });
    arr(o.dots).forEach(function (d) {
      if (num(d.x) && num(d.y)) g += '<circle data-key="' + d.key + '" class="og-dot ' + d.cls + '" cx="' + f(X(d.x)) + '" cy="' + f(Y(d.y)) + '" r="4.6"/>';
    });
    return g + "</svg>";
  }

  function phasePanels(t) {
    var h = "";
    var ph = t.phase || {}, ours = arr(ph.ours), pub = arr(ph.published);
    var epochs = arr(t.epochs);
    function series(col) {
      var s = [];
      if (ours.length) s.push({ pts: ours.map(function (p) { return [p[0], p[col]]; }), cls: "og-ours", key: "ours" });
      if (pub.length) s.push({ pts: pub.map(function (p) { return [p[0], p[col]]; }), cls: "og-p0", dash: true, key: "pub0" });
      return s;
    }
    if (ours.length || pub.length) {
      h += linePlot({ series: series(1), xlabel: "orbital phase", ylabel: "sep (mas)", xmin: 0, xmax: 1, label: "Separation against orbital phase" });
      h += linePlot({ series: series(2), xlabel: "orbital phase", ylabel: "PA (deg)", xmin: 0, xmax: 1, ymin: 0, ymax: 360, wrap: 180, label: "Position angle against orbital phase" });
      return h;
    }
    // no orbit: sep and PA against date, ours (solid) and published (hollow)
    var withSep = epochs.filter(function (e) { return num(e.mjd) && num(e.sep); });
    if (!withSep.length) return "";
    function dots(col) {
      var d = withSep.map(function (e) { return { x: e.mjd, y: e[col], cls: "og-ours", key: "eps" }; });
      withSep.forEach(function (e) { if (e.pub && num(e.pub[col])) d.push({ x: e.mjd, y: e.pub[col], cls: "og-hollow og-p0", key: "pubep" }); });
      return d;
    }
    h += linePlot({ series: [], dots: dots("sep"), xlabel: "MJD", ylabel: "sep (mas)", xprec: 6, label: "Separation against MJD" });
    h += linePlot({ series: [], dots: dots("pa"), xlabel: "MJD", ylabel: "PA (deg)", xprec: 6, ymin: 0, ymax: 360, label: "Position angle against MJD" });
    return h;
  }

  // ---- parameter table ------------------------------------------------
  function table(t) {
    var rows = arr(t.table);
    if (!rows.length) return '<p class="og-muted">No parameter comparison available' + (t.status === "awaiting" ? " until the fit lands" : "") + ".</p>";
    var h = '<div class="og-tw"><table class="og-table"><thead><tr><th>Quantity</th><th>virgil</th><th>Published</th><th>Pull</th><th>Status</th></tr></thead><tbody>';
    rows.forEach(function (r) {
      var st = String(r.status || "").toUpperCase();
      var cls = st === "PASS" || st === "AGREES" ? "og-ok" : st ? "og-bad" : "";
      function pm(v, e) { return fmt(v) + (num(e) ? " ± " + fmt(e) : ""); }
      h += "<tr><td>" + esc(r.quantity) + "</td><td>" + pm(r.ours, r.err) + "</td><td>" + pm(r.pub, r.pub_err) + "</td><td>" + (num(r.pull) ? fmt(Math.round(r.pull * 100) / 100) + "σ" : "–") + '</td><td class="' + cls + '">' + esc(r.status) + "</td></tr>";
    });
    return '<details class="og-tabd"' + (rows.length <= 8 ? " open" : "") + "><summary>" + rows.length + " rows</summary>" + h + "</tbody></table></div></details>";
  }

  // the static per-target tables below the gallery: collapse them too
  function collapseStatic() {
    document.querySelectorAll(".md-typeset table").forEach(function (tb) {
      var w = tb.closest(".og") || tb.closest("details") ;
      if (w) return;
      var box = tb.closest(".md-typeset__scrollwrap") || tb;
      var d = document.createElement("details"); d.className = "og-tabd";
      var sm = document.createElement("summary"); sm.textContent = "Parameter table (" + tb.rows.length + " rows)";
      box.parentNode.insertBefore(d, box); d.appendChild(sm); d.appendChild(box);
    });
  }

  // ---- detail ---------------------------------------------------------
  function detail(root, t, back) {
    var h = '<div class="og-detail"><button type="button" class="og-back">← Back to gallery</button>' +
      '<div class="og-dhead"><h2 class="og-dtitle">' + esc(t.target) + '</h2><span class="og-badge og-b-' + esc(t.status) + '">' + esc(statusText(t.status)) + "</span></div>" +
      '<div class="og-muted">' + esc(t.instrument) + (t.kind === "positions" ? " · detection positions, no orbit fitted" : "") + "</div>";
    if (t.status === "awaiting") h += '<p class="og-note">The fit for this target has not been run yet; the published reference is shown when available.</p>';
    var sky = skyPlot(t);
    // legend
    var leg = [];
    if (t.ours && pts(t.ours.path).length) leg.push(["ours", t.ours.label || "ours", "og-ours", false]);
    if (arr(t.epochs).some(function (e) { return num(e.dra); })) leg.push(["eps", "epochs (1σ)", "og-ours og-sw-dot", false]);
    arr(t.published).forEach(function (p, i) { if (pts(p.path).length) leg.push(["pub" + i, p.label || "published", pubClass(i) + " og-sw-dash", false]); });
    if (sky.hasPub) leg.push(["pubep", "published epochs", "og-p0 og-sw-hollow", false]);
    arr(t.alternatives).forEach(function (a, i) {
      if (pts(a.path).length) leg.push(["alt" + i, (a.label || "mode") + (num(a.dloss) ? " (Δloss " + fmt(a.dloss) + ")" : ""), "og-alt", false]);
    });
    h += '<div class="og-legend" role="group" aria-label="Toggle curves">' + leg.map(function (l) {
      return '<label class="og-tog"><input type="checkbox" checked data-key="' + esc(l[0]) + '"><span class="og-sw ' + l[2] + '"></span>' + esc(l[1]) + "</label>";
    }).join("") + "</div>";
    h += '<div class="og-plots"><div class="og-skywrap">' + sky.html + '</div><div class="og-phase">' + phasePanels(t) + "</div></div>";
    h += '<h3>Parameters against the paper</h3>' + table(t);
    if (arr(t.alternatives).length) {
      h += '<details class="og-alts"><summary>Alternative modes (' + t.alternatives.length + ")</summary><ul>" + t.alternatives.map(function (a) {
        var el = a.elements || {};
        return "<li><strong>" + esc(a.label) + "</strong>" + (num(a.dloss) ? " (Δloss " + fmt(a.dloss) + ")" : "") + ": " +
          Object.keys(el).map(function (k) { return esc(k) + " = " + fmt(el[k]); }).join(", ") + "</li>";
      }).join("") + "</ul></details>";
    }
    if (t.paper && (t.paper.cite || t.paper.arxiv)) {
      h += '<h3>Reference</h3><p class="og-cite">' + esc(t.paper.cite) + (t.paper.arxiv ? " " + chip(t) : "") + "</p>";
    }
    var pv = t.provenance || {};
    var pk = Object.keys(pv).filter(function (k) { return pv[k]; });
    if (pk.length) h += '<p class="og-muted og-prov">Provenance: ' + pk.map(function (k) { return esc(k.replace(/_/g, " ")) + " <code>" + esc(String(pv[k]).slice(0, 10)) + "</code>"; }).join(" · ") + "</p>";
    h += "</div>";
    root.innerHTML = h;
    root.querySelector(".og-back").addEventListener("click", back);

    // toggles
    root.querySelectorAll(".og-tog input").forEach(function (cb) {
      cb.addEventListener("change", function () {
        var k = cb.getAttribute("data-key");
        root.querySelectorAll('svg [data-key="' + k + '"]').forEach(function (el) { el.style.display = cb.checked ? "" : "none"; });
      });
    });

    // tooltips on epochs
    var tip = document.createElement("div");
    tip.className = "og-tip"; tip.hidden = true;
    root.appendChild(tip);
    function show(ev, e) {
      tip.innerHTML = "<strong>MJD " + fmt(e.mjd) + "</strong><br>sep " + fmt(e.sep) + " mas<br>PA " + fmt(e.pa) + "°<br>ΔRA " + fmt(e.dra) + ", ΔDec " + fmt(e.ddec) + " mas";
      tip.hidden = false;
      var rr = root.getBoundingClientRect(), tr = ev.target.getBoundingClientRect();
      var x = (ev.clientX != null && ev.type !== "focus" ? ev.clientX : tr.left + tr.width / 2) - rr.left;
      var y = (ev.clientY != null && ev.type !== "focus" ? ev.clientY : tr.top) - rr.top;
      tip.style.left = Math.max(4, Math.min(x + 10, rr.width - tip.offsetWidth - 4)) + "px";
      tip.style.top = (y + 14) + "px";
    }
    root.querySelectorAll(".og-hit").forEach(function (c) {
      var e = t.epochs[Number(c.getAttribute("data-i"))];
      ["mouseenter", "mousemove", "focus", "click"].forEach(function (n) { c.addEventListener(n, function (ev) { show(ev, e); }); });
      ["mouseleave", "blur"].forEach(function (n) { c.addEventListener(n, function () { tip.hidden = true; }); });
    });
  }

  // ---- wiring ---------------------------------------------------------
  function init() {
    var root = document.querySelector(".og");
    if (root) collapseStatic();
    if (!root || root.getAttribute("data-og-init")) return;
    root.setAttribute("data-og-init", "1");
    var holder = document.getElementById("og-data"), data = null;
    try { data = holder && JSON.parse(holder.textContent); } catch (err) { data = null; }
    if (!data || !Array.isArray(data.targets) || !data.targets.length) return;   // degrade: leave the static page

    function bySlug(s) { return data.targets.filter(function (t) { return t.slug === s; })[0]; }
    function url(slug) { return location.pathname + location.search + (slug ? "#" + slug : ""); }
    function render() {
      var t = bySlug(decodeURIComponent((location.hash || "").slice(1)));
      if (t) {
        detail(root, t, function () { history.pushState(null, "", url("")); render(); });
        root.classList.add("og-in-detail");
        if (root.getBoundingClientRect().top < 0) root.scrollIntoView();
      } else {
        root.classList.remove("og-in-detail");
        gallery(root, data, function (slug) { history.pushState(null, "", url(slug)); render(); });
      }
    }
    root.classList.add("og-ready");
    window.__ogRender = render;
    render();
  }
  window.addEventListener("popstate", function () { if (window.__ogRender && document.querySelector(".og[data-og-init]")) window.__ogRender(); });
  window.addEventListener("hashchange", function () { if (window.__ogRender && document.querySelector(".og[data-og-init]")) window.__ogRender(); });
  if (typeof document$ !== "undefined") document$.subscribe(init); else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
