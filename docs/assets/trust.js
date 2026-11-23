// The Trust page: select a part or a pipeline to see its cone (everything it relies on, down to the
// ground) and how it was checked. scripts/trust.py embeds the data as #vt-data; the cones and their
// sentences come from scripts/trust_cone.py, so this file only draws them.
(function () {
  var onHash = null;

  function init() {
    var holder = document.getElementById("vt-data");
    var panel = document.getElementById("vt-detail");
    if (!holder || !panel) return;
    var data = JSON.parse(holder.textContent);
    var nodes = data.nodes, verdicts = data.verdicts, roots = data.roots, C = data.cone;
    var map = document.querySelector(".vt-map");
    var sky = document.querySelector(".vt-sky");
    var cone = document.getElementById("vt-cone");
    var current = null;
    function esc(s) { var d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; }
    function fmt(v) {
      if (v == null || typeof v !== "number") return esc(v);
      return (v !== 0 && (Math.abs(v) < 1e-2 || Math.abs(v) >= 1e4)) ? v.toExponential(1) : String(Number(v.toPrecision(3)));
    }
    function sel(id) { return '[data-node="' + CSS.escape(id) + '"]'; }
    function all(q, fn) { Array.prototype.forEach.call(document.querySelectorAll(q), fn); }

    // ---- the detail panel in the list fold (how the part was checked)
    function detail(n) {
      var v = verdicts[n.verdict];
      var html = "<h4>" + esc(n.id) + "</h4><div><strong>" + esc(v[0]) + "</strong>: " + esc(v[1]) + ".</div>";
      html += '<div><button type="button" class="vt-cone-jump">Show its cone ↑</button></div>';
      if (n.doc) html += '<div class="vt-read"><a href="' + esc(n.doc) + '">Read how it was checked →</a></div>';
      if (n.findings.length) html += "<div>Open: " + n.findings.map(function (f) { return '<a href="#' + esc(f) + '">' + esc(f) + "</a>"; }).join(", ") + "</div>";
      if (n.because.length) html += "<div>" + (n.verdict.indexOf("relies") === 0 ? "Relies on " : "Also relies on ") + n.because.map(function (b) {
        return "<code>" + esc(b.id) + "</code>" + (b.findings.length ? " (" + b.findings.map(function (f) { return '<a href="#' + esc(f) + '">' + esc(f) + "</a>"; }).join(", ") + ")" : "");
      }).join(", ") + (n.verdict.indexOf("relies") === 0 ? "; it counts as verified once that is fixed." : "; fixing that alone would not verify it.") + "</div>";
      var rs = n.strong.concat(n.weak);
      html += "<div>Checked against: " + (rs.length ? rs.map(function (r) { return '<span class="vt-root">' + esc(roots[r] || r) + "</span>"; }).join("") : "nothing yet") + "</div>";
      if (n.changed && n.changed.length) html += '<div class="vt-muted">virgil has changed ' + n.changed.map(function (f) { return "<code>" + esc(f) + "</code>"; }).join(", ") + " since this evidence.</div>";
      var checks = n.checks.filter(function (c) { return c.kind !== "guard" && c.outcome !== "skipped"; });
      if (checks.length) {
        html += '<details class="vt-source"><summary>Tests (' + checks.length + ")</summary>";
        html += '<ul class="vt-checks">' + checks.slice(0, 12).map(function (c) {
          var out = c.outcome === "passed" ? "" : " <em>(" + esc(c.outcome) + ")</em>";
          var val = c.headline ? " — " + esc(c.headline) + " = " + fmt(c.value) : "";
          var name = c.url ? '<a href="' + esc(c.url) + '">' + esc(c.name) + "</a>" : esc(c.name);
          return "<li>" + name + out + val + (c.doc ? '<div class="vt-muted">' + esc(c.doc) + "</div>" : "") + "</li>";
        }).join("") + "</ul>";
        if (checks.length > 12) html += '<div class="vt-muted">and ' + (checks.length - 12) + " more in the table below.</div>";
        html += "</details>";
      }
      panel.innerHTML = html;
    }

    // ---- the cone panel
    var body = null, pick = null;
    function buildCone() {
      if (!cone) return;
      var opts = Object.keys(C.pipelines).map(function (k) {
        return '<option value="' + esc(k) + '">' + esc(C.pipelines[k].title) + " (" + esc(C.pipelines[k].status) + ")</option>";
      }).join("");
      cone.innerHTML = '<h3 id="vt-cone-h">The cone of one part or pipeline</h3>' +
        '<div class="vt-cone-bar"><label>Pipeline <select class="vt-cone-pick"><option value="">Choose a pipeline…</option>' + opts + "</select></label>" +
        '<button type="button" class="vt-cone-clear" hidden>Clear</button></div>' +
        '<div class="vt-cone-live" aria-live="polite"><p class="vt-muted vt-cone-hint">Select a square in the skyline, a part in the list below or a pipeline here. ' +
        "The panel shows everything it relies on, down to the ground.</p></div>" +
        '<div class="vt-cone-key vt-muted" hidden></div>' +
        '<div class="vt-cone-scroll" role="region" aria-label="Cone drawing; scrolls sideways when wide" tabindex="0" hidden></div>';
      body = {
        live: cone.querySelector(".vt-cone-live"), key: cone.querySelector(".vt-cone-key"),
        scroll: cone.querySelector(".vt-cone-scroll"), clear: cone.querySelector(".vt-cone-clear"),
      };
      pick = cone.querySelector(".vt-cone-pick");
      pick.addEventListener("change", function () { if (pick.value) select(pick.value, {scroll: false}); else clear(); });
      body.clear.addEventListener("click", clear);
    }
    function coneIds(id) { return C.cones[id].map(function (i) { return C.order[i]; }); }
    function trunc(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }
    function lettersOf(n) {
      var seen = {};
      n.strong.forEach(function (r) { seen[C.letters[r] || (r.indexOf("golden") === 0 ? "G" : "?")] = true; });
      return C.letter_order.split("").filter(function (l) { return seen[l]; }).concat(seen["?"] ? ["?"] : []);
    }
    var BH = 48, RH = 76, PAD = 14, LINE = 58, DGAP = 30;   // box height, pipeline row, margin, one line of boxes, gap between depths

    function drawCone(id) {
      var pipe = C.pipelines[id] && !nodes[id] ? C.pipelines[id] : null;
      var ids = coneIds(id);
      var members = {}; ids.forEach(function (m) { members[m] = true; });
      var rows = {};
      ids.forEach(function (m) { (rows[nodes[m].depth] = rows[nodes[m].depth] || []).push(m); });
      var depths = Object.keys(rows).map(Number).sort(function (a, b) { return b - a; });
      // the drawing is as wide as its frame, never wider: a depth with more parts than fit across wraps onto further lines
      var avail = Math.floor(body.scroll.clientWidth - 14) || 900;
      var narrow = avail < 560, LEFT = narrow ? 26 : 56, GX = narrow ? 8 : 12, BW = 156;
      if (avail - LEFT - PAD < 2 * (BW + GX)) BW = Math.max(96, Math.floor((avail - LEFT - PAD) / 2) - GX);
      var cap = Math.max(1, Math.floor((avail - LEFT - PAD) / (BW + GX)));
      var chars = Math.max(8, Math.floor((BW - 22) / 6.2));
      var Wc = Math.max(avail, LEFT + PAD + 2 * (BW + GX)), cx = LEFT + (Wc - LEFT) / 2;
      var ydep = {}, yy = PAD + (pipe ? RH : 0), lines = {};
      depths.forEach(function (d) { lines[d] = Math.ceil(rows[d].length / cap); ydep[d] = yy; yy += (lines[d] - 1) * LINE + BH + DGAP; });
      var Hc = yy - DGAP + PAD + 8;
      var x = {}, ny = {};     // x and y (top) of each box
      var parents = {};
      ids.forEach(function (m) { nodes[m].depends.forEach(function (b) { if (members[b]) (parents[b] = parents[b] || []).push(m); }); });
      function at(m) { return x[m] != null ? x[m] : null; }
      if (pipe) { x["pipe"] = cx; }
      function pe(m) { return pipe && pipe.steps.indexOf(m) >= 0; }
      depths.forEach(function (d) {
        var row = rows[d].slice();
        var bary = function (m) {
          var ps = (parents[m] || []).map(at).filter(function (v) { return v != null; });
          if (pe(m)) ps.push(cx);
          return ps.length ? ps.reduce(function (s, v) { return s + v; }, 0) / ps.length : cx;
        };
        var b = {}; row.forEach(function (m) { b[m] = bary(m); });
        row.sort(function (p, q) { return (b[p] - b[q]) || (nodes[p].label < nodes[q].label ? -1 : 1); });
        // fill the lines evenly (7, 7, 6 rather than 12, 1), left to right in barycentre order
        var per = Math.ceil(row.length / lines[d]);
        row.forEach(function (m, i) {
          var ln = Math.floor(i / per), first = ln * per, count = Math.min(per, row.length - first);
          x[m] = cx + ((i - first) - (count - 1) / 2) * (BW + GX);
          ny[m] = ydep[d] + ln * LINE;
        });
      });
      // edges, first so that the boxes sit on top
      var edges = [], seen = {};
      function edge(a, b, ya, yb) {
        var k = a + ">" + b; if (seen[k] || a === b) return; seen[k] = 1;
        var x1 = x[a], x2 = x[b], my = (ya + yb) / 2;
        edges.push('<path d="M' + x1 + "," + ya + " C" + x1 + "," + my + " " + x2 + "," + my + " " + x2 + "," + yb + '"/>');
      }
      ids.forEach(function (m) {
        nodes[m].depends.forEach(function (b) { if (members[b]) edge(m, b, ny[m] + BH, ny[b]); });
      });
      if (pipe) pipe.steps.forEach(function (s) { if (members[s]) edge("pipe", s, PAD + 34, ny[s]); });
      var out = ['<svg class="vt-cone-svg" viewBox="0 0 ' + Wc + " " + Hc + '" width="' + Wc + '" height="' + Hc +
        '" role="group" aria-label="Cone of ' + esc(pipe ? pipe.title : nodes[id].label) + ": " + ids.length + " parts in " + depths.length +
        ' depths, from the selected part at the top to the ground at the bottom">', '<g class="vt-cone-edges">' + edges.join("") + "</g>"];
      depths.forEach(function (d) {
        out.push('<text class="vt-cone-d" x="' + (LEFT - 8) + '" y="' + (ydep[d] + BH / 2 + 4) + '" text-anchor="end">' + (narrow ? d : "depth " + d) + "</text>");
        if (lines[d] > 1) out.push('<line class="vt-cone-span" x1="' + (LEFT - 4) + '" x2="' + (LEFT - 4) + '" y1="' + ydep[d] + '" y2="' + (ydep[d] + (lines[d] - 1) * LINE + BH) + '"/>');
        rows[d].forEach(function (mid) {
          var n = nodes[mid], bx = x[mid] - BW / 2, by = ny[mid], v = verdicts[n.verdict][0], L = lettersOf(n);
          var word = (C.tag[n.verdict] || n.verdict), sel = mid === id ? " sel" : "";
          out.push('<a class="vt-cone-box' + sel + '" href="#' + encodeURIComponent(mid) + '" data-cone-node="' + esc(mid) + '" aria-label="' +
            esc(n.label + ": " + v + ". " + (L.length ? "Checked against " + L.map(function (l) { return C.letter_names[l] || "an unlisted root"; }).join(", ") : "No root yet")) + '">' +
            "<title>" + esc(n.id + " · " + v + " · depth " + n.depth + " · " + (n.strong.length ? "checked against " + n.strong.map(function (r) { return roots[r] || r; }).join(", ") : "no counted root")) + "</title>" +
            '<rect class="bx" x="' + bx + '" y="' + by + '" width="' + BW + '" height="' + BH + '" rx="5"/>' +
            '<rect class="st k-' + n.verdict + '" x="' + bx + '" y="' + by + '" width="7" height="' + BH + '" rx="3"/>' +
            '<text class="l1" x="' + (bx + 14) + '" y="' + (by + 15) + '"><tspan class="ic">' + esc(data.cone.icons[n.verdict] || "") + "</tspan> " + esc(trunc(n.label, chars)) + "</text>" +
            '<text class="l2" x="' + (bx + 14) + '" y="' + (by + 28) + '">' + esc(trunc(word, chars + 4)) + "</text>" +
            '<text class="l3" x="' + (bx + 14) + '" y="' + (by + 41) + '">' + (L.length ? esc(L.join(" ")) : "no root") + "</text></a>");
        });
      });
      if (pipe) {
        out.push('<g class="vt-cone-pipe"><rect class="bx" x="' + (cx - 130) + '" y="' + PAD + '" width="260" height="34" rx="17"/><text x="' + cx + '" y="' + (PAD + 21) + '" text-anchor="middle">' +
          esc(trunc(pipe.title, 34)) + "</text></g>");
      }
      out.push('<rect class="vt-cone-ground" x="' + (LEFT - 4) + '" y="' + (Hc - 5) + '" width="' + (Wc - LEFT - 4) + '" height="3"/>');
      out.push("</svg>");
      return out.join("");
    }

    function keyHtml(id) {
      var counts = {}, used = {};
      coneIds(id).forEach(function (m) {
        counts[nodes[m].verdict] = (counts[nodes[m].verdict] || 0) + 1;
        lettersOf(nodes[m]).forEach(function (l) { used[l] = true; });
      });
      var vs = Object.keys(verdicts).filter(function (v) { return counts[v]; }).map(function (v) {
        return '<span class="vt-key"><span class="vt-sw vt-' + v + '" aria-hidden="true">' + esc(data.cone.icons[v] || "") + "</span>" + esc(verdicts[v][0]) + " (" + counts[v] + ")</span>";
      }).join("");
      var ls = C.letter_order.split("").concat(["?"]).filter(function (l) { return used[l]; }).map(function (l) {
        return "<b>" + l + "</b> " + (C.letter_names[l] || "an unlisted root");
      }).join(" · ");
      return '<div class="vt-legend">' + vs + "</div>" + (ls ? '<div>Root letters in this cone: ' + ls + ".</div>" : "");
    }

    function draw(id) {   // the frame must be visible to be measured: the drawing is made to fit its width
      body.scroll.hidden = false;
      body.scroll.innerHTML = drawCone(id);
    }
    var resizeTimer = null;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () { if (current && cone && !body.scroll.hidden) draw(current); }, 150);
    });
    function showCone(id) {
      var pipe = !nodes[id];
      var n = pipe ? null : nodes[id], p = pipe ? C.pipelines[id] : null;
      var head = pipe
        ? "<h4>" + esc(p.title) + ' <span class="vt-badge">pipeline · ' + esc(p.status) + "</span></h4>"
        : "<h4><code>" + esc(n.id) + '</code> <span class="vt-badge"><span aria-hidden="true">' + esc(C.icons[n.verdict] || "") + "</span> " + esc(verdicts[n.verdict][0]) + "</span></h4>";
      var link = pipe ? p.page : n.doc;
      body.live.innerHTML = head + "<p>" + esc(C.sentences[id]) + "</p>" +
        (link ? '<p><a href="' + esc(link) + '">' + (pipe ? "Read about this pipeline" : "Read how it was checked") + " →</a></p>" : "");
      body.key.innerHTML = keyHtml(id); body.key.hidden = false;
      draw(id);
      body.clear.hidden = false;
      pick.value = pipe ? id : "";
    }

    // ---- selection
    function mark(id) {
      all(".vt-sel, .vt-up, .vt-down, .vt-in", function (el) { el.classList.remove("vt-sel", "vt-up", "vt-down", "vt-in"); });
      if (sky) sky.classList.remove("vt-dim");
      if (!id) return;
      var members = coneIds(id);
      var chip = ".vt-map [data-node], .vt-step[data-node]";
      if (nodes[id]) {
        all(chip, function (el) { var k = el.getAttribute("data-node"); if (k === id) el.classList.add("vt-sel"); });
        nodes[id].depends.forEach(function (d) { all(".vt-map " + sel(d) + ", .vt-step" + sel(d), function (el) { el.classList.add("vt-down"); }); });
        nodes[id].dependents.forEach(function (d) { all(".vt-map " + sel(d) + ", .vt-step" + sel(d), function (el) { el.classList.add("vt-up"); }); });
        if (map) map.classList.add("vt-dim");
      } else if (map) map.classList.remove("vt-dim");
      if (sky) {
        sky.classList.add("vt-dim");
        members.forEach(function (m) { all(".vt-sky " + sel(m), function (el) { el.classList.add("vt-in"); }); });
        if (nodes[id]) all(".vt-sky " + sel(id), function (el) { el.classList.add("vt-sel"); });
      }
    }
    function select(id, o) {
      o = o || {};
      if (!C.cones[id]) return false;
      current = id;
      if (nodes[id]) {
        detail(nodes[id]);
        if (o.fold) { var fold = document.getElementById("parts-list"); if (fold) fold.open = true; }
      }
      mark(id);
      if (cone) showCone(id);
      if (o.hash !== false) {
        try { history.replaceState(null, "", "#" + encodeURIComponent(id)); } catch (err) { /* a file: page cannot */ }
      }
      if (o.scroll && cone) cone.scrollIntoView({block: o.block || "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"});
      return true;
    }
    function clear() {
      if (current && decodeURIComponent(location.hash.slice(1)) === current) {
        try { history.replaceState(null, "", location.pathname + location.search); } catch (err) { /* ignore */ }
      }
      current = null; mark(null);
      if (cone) {
        body.live.innerHTML = '<p class="vt-muted vt-cone-hint">Select a square in the skyline, a part in the list below or a pipeline here.</p>';
        body.key.hidden = true; body.scroll.hidden = true; body.clear.hidden = true; pick.value = "";
      }
    }
    function fromHash(first) {
      var h; try { h = decodeURIComponent(location.hash.slice(1)); } catch (err) { return; }
      if (!h || h === current || !C.cones[h]) return;
      select(h, {hash: false, scroll: !first, block: "start"});
      // on load the page is still settling (images above the panel are lazy): scroll once it has
      if (first && cone) {
        var go = function () { if (current === h) cone.scrollIntoView({block: "start"}); };
        if (document.readyState === "complete") setTimeout(go, 50); else window.addEventListener("load", function () { setTimeout(go, 50); }, {once: true});
      }
    }

    // ---- wiring
    buildCone();
    all(".vt [data-node]", function (el) {
      var inSky = !!el.closest(".vt-sky"), inMap = !!el.closest(".vt-map");
      if (inSky && el.tagName.toLowerCase() === "g") { el.setAttribute("tabindex", "0"); el.setAttribute("role", "button"); }
      el.addEventListener("click", function (ev) {
        if (inSky) ev.preventDefault();
        select(el.getAttribute("data-node"), inMap ? {fold: true} : {scroll: true});
      });
      el.addEventListener("keydown", function (ev) {
        if ((ev.key === "Enter" || ev.key === " ") && el.tagName.toLowerCase() === "g") { ev.preventDefault(); el.dispatchEvent(new MouseEvent("click", {bubbles: true})); }
      });
    });
    panel.addEventListener("click", function (ev) {
      if (ev.target.classList && ev.target.classList.contains("vt-cone-jump") && cone) cone.scrollIntoView({block: "start"});
    });
    if (cone) {
      cone.addEventListener("click", function (ev) {
        var box = ev.target.closest && ev.target.closest("[data-cone-node]");
        if (box) { ev.preventDefault(); select(box.getAttribute("data-cone-node"), {}); }
      });
    }
    if (onHash) window.removeEventListener("hashchange", onHash);
    onHash = function () { fromHash(false); };
    window.addEventListener("hashchange", onHash);
    fromHash(true);
  }
  if (typeof document$ !== "undefined") document$.subscribe(init); else document.addEventListener("DOMContentLoaded", init);
})();
