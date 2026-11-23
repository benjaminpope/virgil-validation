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
    var current = null, expanded = {};
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
    var BW = 156, BH = 48, GX = 12, RH = 76, PAD = 14, LEFT = 50, MAXROW = 12;

    function drawCone(id) {
      var pipe = C.pipelines[id] && !nodes[id] ? C.pipelines[id] : null;
      var ids = coneIds(id);
      var members = {}; ids.forEach(function (m) { members[m] = true; });
      var rows = {};
      ids.forEach(function (m) { (rows[nodes[m].depth] = rows[nodes[m].depth] || []).push(m); });
      var depths = Object.keys(rows).map(Number).sort(function (a, b) { return b - a; });
      var rank = {}; Object.keys(verdicts).forEach(function (v, i) { rank[v] = v === "verified" ? 99 : i; });
      var more = {}, vis = {};   // hidden ids of a collapsed row; visible ids of each row
      depths.forEach(function (d) {
        var row = rows[d].slice();
        if (row.length > MAXROW && !expanded[d]) {
          row.sort(function (a, b) { return (rank[nodes[a].verdict] - rank[nodes[b].verdict]) || (nodes[a].label < nodes[b].label ? -1 : 1); });
          more[d] = row.slice(MAXROW - 1); row = row.slice(0, MAXROW - 1);
        }
        vis[d] = row;
      });
      var widest = Math.max.apply(null, depths.map(function (d) { return vis[d].length + (more[d] ? 1 : 0); }).concat([pipe ? 1 : 0, 2]));
      var Wc = LEFT + PAD + widest * (BW + GX), cx = LEFT + (Wc - LEFT) / 2;
      var y = {}, yy = PAD + (pipe ? RH : 0);
      depths.forEach(function (d) { y[d] = yy; yy += RH; });
      var Hc = yy - RH + BH + PAD + 8;
      var x = {}, where = {};     // x of each drawn box; where[id] is the box that stands for it
      var parents = {};
      ids.forEach(function (m) { nodes[m].depends.forEach(function (b) { if (members[b]) (parents[b] = parents[b] || []).push(m); }); });
      function at(m) { return where[m] != null ? x[where[m]] : null; }
      if (pipe) { x["pipe"] = cx; }
      depths.forEach(function (d) {
        var row = vis[d].slice();
        var bary = function (m) {
          var ps = (parents[m] || []).map(at).filter(function (v) { return v != null; });
          if (pe(m) && pipe) ps.push(cx);
          return ps.length ? ps.reduce(function (s, v) { return s + v; }, 0) / ps.length : cx;
        };
        function pe(m) { return pipe && pipe.steps.indexOf(m) >= 0; }
        row.sort(function (a, b) { return (bary(a) - bary(b)) || (nodes[a].label < nodes[b].label ? -1 : 1); });
        var items = row.map(function (m) { return {id: m}; });
        if (more[d]) items.push({more: d});
        items.forEach(function (it, j) {
          var key = it.more != null ? "more:" + d : it.id;
          x[key] = cx + (j - (items.length - 1) / 2) * (BW + GX);
          if (it.id) where[it.id] = key; else more[d].forEach(function (m) { where[m] = key; });
        });
        vis[d] = items;
      });
      // edges, first so that the boxes sit on top
      var edges = [], seen = {};
      function edge(a, b, ya, yb) {   // a, b are box keys
        var k = a + ">" + b; if (seen[k] || a === b) return; seen[k] = 1;
        var x1 = x[a], x2 = x[b], my = (ya + yb) / 2;
        edges.push('<path d="M' + x1 + "," + ya + " C" + x1 + "," + my + " " + x2 + "," + my + " " + x2 + "," + yb + '"/>');
      }
      ids.forEach(function (m) {
        nodes[m].depends.forEach(function (b) { if (members[b]) edge(where[m], where[b], y[nodes[m].depth] + BH, y[nodes[b].depth]); });
      });
      if (pipe) pipe.steps.forEach(function (s) { if (members[s]) edge("pipe", where[s], PAD + 34, y[nodes[s].depth]); });
      var out = ['<svg class="vt-cone-svg" data-cx="' + cx + '" viewBox="0 0 ' + Wc + " " + Hc + '" width="' + Wc + '" height="' + Hc +
        '" role="group" aria-label="Cone of ' + esc(pipe ? pipe.title : nodes[id].label) + ": " + ids.length + " parts in " + depths.length +
        ' rows, from the selected part at the top to the ground at the bottom">', '<g class="vt-cone-edges">' + edges.join("") + "</g>"];
      depths.forEach(function (d) {
        out.push('<text class="vt-cone-d" x="' + (LEFT - 8) + '" y="' + (y[d] + BH / 2 + 4) + '" text-anchor="end">depth ' + d + "</text>");
        vis[d].forEach(function (it) {
          if (it.more != null) {
            var hid = more[it.more], bx = x["more:" + it.more] - BW / 2;
            out.push('<g class="vt-cone-more" tabindex="0" role="button" data-more="' + it.more + '" aria-label="Show ' + hid.length + " more parts at depth " + it.more + '">' +
              "<title>" + hid.length + " more parts at depth " + it.more + " (the unverified ones are shown first)</title>" +
              '<rect class="bx" x="' + bx + '" y="' + y[it.more] + '" width="' + BW + '" height="' + BH + '" rx="5"/>' +
              '<text x="' + (bx + BW / 2) + '" y="' + (y[it.more] + BH / 2 + 4) + '" text-anchor="middle">+' + hid.length + " more</text></g>");
            return;
          }
          var n = nodes[it.id], bx = x[it.id] - BW / 2, by = y[d], v = verdicts[n.verdict][0], L = lettersOf(n);
          var word = (C.short[n.verdict] || n.verdict), sel = it.id === id ? " sel" : "";
          out.push('<a class="vt-cone-box' + sel + '" href="#' + encodeURIComponent(it.id) + '" data-cone-node="' + esc(it.id) + '" aria-label="' +
            esc(n.label + ": " + v + ". " + (L.length ? "Checked against " + L.map(function (l) { return C.letter_names[l] || "an unlisted root"; }).join(", ") : "No root yet")) + '">' +
            "<title>" + esc(n.id + " · " + v + " · depth " + n.depth + " · " + (n.strong.length ? "checked against " + n.strong.map(function (r) { return roots[r] || r; }).join(", ") : "no counted root")) + "</title>" +
            '<rect class="bx" x="' + bx + '" y="' + by + '" width="' + BW + '" height="' + BH + '" rx="5"/>' +
            '<rect class="st k-' + n.verdict + '" x="' + bx + '" y="' + by + '" width="7" height="' + BH + '" rx="3"/>' +
            '<text class="l1" x="' + (bx + 14) + '" y="' + (by + 15) + '"><tspan class="ic">' + esc(data.cone.icons[n.verdict] || "") + "</tspan> " + esc(trunc(n.label, 22)) + "</text>" +
            '<text class="l2" x="' + (bx + 14) + '" y="' + (by + 28) + '">' + esc(trunc(word, 28)) + "</text>" +
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

    function draw(id) {   // the drawing is wider than a phone: scroll its container so the top part is in view
      body.scroll.innerHTML = drawCone(id); body.scroll.hidden = false;
      var svg = body.scroll.firstChild;
      body.scroll.scrollLeft = Math.max(0, +svg.getAttribute("data-cx") - body.scroll.clientWidth / 2);
    }
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
      current = id; expanded = {};
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
        if (box) { ev.preventDefault(); select(box.getAttribute("data-cone-node"), {}); return; }
        var m = ev.target.closest && ev.target.closest("[data-more]");
        if (m) { expanded[m.getAttribute("data-more")] = true; draw(current); var first = body.scroll.querySelector("a"); if (first) first.focus({preventScroll: true}); }
      });
      cone.addEventListener("keydown", function (ev) {
        var m = ev.target.closest && ev.target.closest("[data-more]");
        if (m && (ev.key === "Enter" || ev.key === " ")) { ev.preventDefault(); m.dispatchEvent(new MouseEvent("click", {bubbles: true})); }
      });
    }
    if (onHash) window.removeEventListener("hashchange", onHash);
    onHash = function () { fromHash(false); };
    window.addEventListener("hashchange", onHash);
    fromHash(true);
  }
  if (typeof document$ !== "undefined") document$.subscribe(init); else document.addEventListener("DOMContentLoaded", init);
})();
