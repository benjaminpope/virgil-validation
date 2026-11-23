// The Trust page: select a part or a pipeline and the skyline lights its cone (everything it relies on, down to the
// ground), with a panel under the figure. scripts/trust.py embeds the data as #vt-data; the cones, paths and their
// sentences come from scripts/trust_cone.py, so this file only draws them.
(function () {
  var onHash = null;

  function init() {
    var holder = document.getElementById("vt-data");
    var panel = document.getElementById("vt-detail");
    if (!holder || !panel) return;
    var data = JSON.parse(holder.textContent);
    var nodes = data.nodes, verdicts = data.verdicts, roots = data.roots, C = data.cone;
    function rootName(r) { return roots[r] || (r.indexOf("golden:") === 0 ? "reference code: " + r.slice(7) : r); }
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
      html += '<div><button type="button" class="vt-cone-jump">Show it on the skyline ↑</button></div>';
      if (n.doc) html += '<div class="vt-read"><a href="' + esc(n.doc) + '">Read how it was checked →</a></div>';
      if (n.findings.length) html += "<div>Open: " + n.findings.map(function (f) { return '<a href="#' + esc(f) + '">' + esc(f) + "</a>"; }).join(", ") + "</div>";
      if (n.because.length) html += "<div>" + (n.verdict.indexOf("relies") === 0 ? "Relies on " : "Also relies on ") + n.because.map(function (b) {
        return "<code>" + esc(b.id) + "</code>" + (b.findings.length ? " (" + b.findings.map(function (f) { return '<a href="#' + esc(f) + '">' + esc(f) + "</a>"; }).join(", ") + ")" : "");
      }).join(", ") + (n.verdict.indexOf("relies") === 0 ? "; it counts as verified once that is fixed." : "; fixing that alone would not verify it.") + "</div>";
      var rs = n.strong.concat(n.weak);
      html += "<div>Checked against: " + (rs.length ? rs.map(function (r) { return '<span class="vt-root">' + esc(rootName(r)) + "</span>"; }).join("") : "nothing yet") + "</div>";
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

    // ---- the lens: the selection lights its cone on the skyline; the panel under the figure says what it means
    var NS = "http://www.w3.org/2000/svg";
    var svg = sky ? sky.querySelector(".vt-sky-scroll svg") : null;
    var overlay = null, pos = {}, idx = {}, inCone = {}, hov = null;
    C.order.forEach(function (id, i) { idx[id] = i; });
    function ids(list) { return list.map(function (i) { return C.order[i]; }); }
    function coneIds(id) { return ids(C.cones[id]); }
    function label(id) { return nodes[id].label || id; }
    function colour(v) { return (C.colours && C.colours[v]) || "currentColor"; }
    if (svg) {
      all(".vt-sky [data-node]", function (el) {
        var r = el.querySelector("rect.sq");
        if (r) pos[el.getAttribute("data-node")] = {cx: +r.getAttribute("x") + 5.5, cy: +r.getAttribute("y") + 5.5};
      });
      overlay = document.createElementNS(NS, "g");
      overlay.setAttribute("class", "vt-lens");
      overlay.setAttribute("aria-hidden", "true");
      svg.appendChild(overlay);
    }
    function curve(a, b) {
      var A = pos[a], B = pos[b];
      if (!A || !B) return "";
      var dy = B.cy - A.cy;
      return "M" + A.cx + "," + A.cy + "C" + A.cx + "," + (A.cy + dy * 0.5) + " " + B.cx + "," + (B.cy - dy * 0.5) + " " + B.cx + "," + B.cy;
    }
    function chainD(list) { return list.slice(1).map(function (x, i) { return curve(list[i], x); }).join(""); }
    function drawLens(id) {
      if (!overlay) return;
      var L = C.lens[id], crit = ids(L.crit), chain = ids(L.chain), h = "";
      if (chain.length > 1) {
        var d = chainD(chain);
        h += '<path class="vt-halo" d="' + d + '"/><path class="vt-long" d="' + d + '"/>';
      }
      if (crit.length > 1) {
        var t = crit[crit.length - 1], dc = chainD(crit), q = pos[t], col = colour(nodes[t].verdict);
        h += '<path class="vt-halo" d="' + dc + '"/><path class="vt-crit" d="' + dc + '" stroke="' + col + '"/>';
        if (q) h += '<circle class="vt-halo" cx="' + q.cx + '" cy="' + q.cy + '" r="10" fill="none"/><circle class="vt-target" cx="' + q.cx + '" cy="' + q.cy + '" r="9" stroke="' + col + '"/>';
      }
      overlay.innerHTML = h + '<g class="vt-hov"></g>';
    }
    function showHover(id) {   // every edge of the cone at a lit square, while it is hovered or focused
      if (!overlay || hov === id) return;
      hov = id;
      var g = overlay.querySelector(".vt-hov");
      if (!g) return;
      var d = "";
      if (id) {
        nodes[id].depends.forEach(function (b) { if (inCone[b]) d += curve(id, b); });
        nodes[id].dependents.forEach(function (a) { if (inCone[a]) d += curve(a, id); });
      }
      g.innerHTML = d ? '<path d="' + d + '"/>' : "";
    }
    // the foundation matrix under the ground line: with a selection, each cell reads cone/all
    var cells = [];
    all(".vt-sky g[data-c]", function (g) { cells.push({g: g, t: g.querySelector("text"), c: g.getAttribute("data-c"), l: g.getAttribute("data-l"), n: g.getAttribute("data-n")}); });
    function foundation(members) {
      var cnt = {};
      members.forEach(function (m) {
        var seen = {};
        nodes[m].strong.forEach(function (r) { seen[C.root_class[r] || "unlisted"] = true; });
        Object.keys(seen).forEach(function (c) { var k = c + "|" + nodes[m].layer; cnt[k] = (cnt[k] || 0) + 1; });
      });
      return cnt;
    }
    function showFoundation(members) {
      var cnt = members ? foundation(members) : null;
      cells.forEach(function (x) {
        var k = cnt ? cnt[x.c + "|" + x.l] || 0 : 0;
        x.g.classList.toggle("vt-hit", !!k);
        x.t.innerHTML = cnt ? (k ? '<tspan class="vt-fc">' + k + "</tspan>" : "") + '<tspan class="vt-fa">' + (k ? "/" : "") + x.n + "</tspan>" : esc(x.n);
      });
    }

    // ---- the panel
    var body = null, pick = null;
    function buildPanel() {
      if (!cone) return;
      var opts = Object.keys(C.pipelines).map(function (k) {
        return '<option value="' + esc(k) + '">' + esc(C.pipelines[k].title) + " (" + esc(C.pipelines[k].status) + ")</option>";
      }).join("");
      cone.innerHTML = '<h3 id="vt-cone-h">What a part rests on</h3>' +
        '<div class="vt-cone-bar"><label>Pipeline <select class="vt-cone-pick"><option value="">Choose a pipeline…</option>' + opts + "</select></label>" +
        '<button type="button" class="vt-cone-clear" hidden>Clear selection</button></div>' +
        '<div class="vt-cone-live" aria-live="polite"><p class="vt-muted vt-cone-hint">Select a square in the skyline, a part in the list below or a pipeline here. ' +
        "The skyline lights everything it relies on, down to the ground.</p></div>";
      body = {live: cone.querySelector(".vt-cone-live"), clear: cone.querySelector(".vt-cone-clear")};
      pick = cone.querySelector(".vt-cone-pick");
      pick.addEventListener("change", function () { if (pick.value) select(pick.value, {}); else clear(); });
      body.clear.addEventListener("click", clear);
    }
    function icon(v) { return '<span class="vt-sw vt-' + v + '" aria-hidden="true">' + esc(C.icons[v] || "") + "</span>"; }
    function link(id) { return '<a href="#' + encodeURIComponent(id) + '">' + esc(label(id)) + "</a>"; }
    function chainHtml(list) {
      var shown = list.length > 7 ? list.slice(0, 3).map(link).concat(["…"], list.slice(-2).map(link)) : list.map(link);
      return shown.join(" → ");
    }
    function fixItem(it) {
      var id = C.order[it[0]], n = nodes[id], note = it[1] === 1 ? "this part" : it[1] === 2 ? "via reference code" : "";
      return '<li><a href="#' + encodeURIComponent(id) + '">' + icon(n.verdict) + '<span class="vt-fx-l">' + esc(label(id)) +
        (n.findings.length ? ' <span class="vt-fx-w">' + esc(n.findings.join(" ")) + "</span>" : "") + '</span><span class="vt-fx-n">blocks ' + C.blocks[it[0]] +
        '</span><span class="vt-fx-w">' + esc(C.short[n.verdict]) + (note ? " · " + note : "") + "</span></a></li>";
    }
    function showCone(id) {
      var pipe = !nodes[id], L = C.lens[id];
      var n = pipe ? null : nodes[id], p = pipe ? C.pipelines[id] : null;
      var head = pipe
        ? "<h4>" + esc(p.title) + ' <span class="vt-badge">pipeline · ' + esc(p.status) + "</span></h4>"
        : "<h4><code>" + esc(n.id) + '</code> <span class="vt-badge">' + icon(n.verdict) + " " + esc(verdicts[n.verdict][0]) + "</span></h4>";
      var link_ = pipe ? p.page : n.doc;
      var left = head + "<p>" + esc(C.sentences[id]) + "</p>" +
        (link_ ? '<p><a href="' + esc(link_) + '">' + (pipe ? "Read about this pipeline" : "Read how it was checked") + " →</a></p>" : "");
      var crit = ids(L.crit), chain = ids(L.chain);
      if (crit.length > 1 || chain.length > 1) left += "<h5>Paths on the figure</h5>";
      if (crit.length > 1) {
        var w = crit[crit.length - 1];
        left += '<p class="vt-path"><svg viewBox="0 0 30 8" width="30" height="8" aria-hidden="true"><path d="M1,4H29" stroke="' + colour(nodes[w].verdict) + '" stroke-width="3" stroke-linecap="round"/></svg>' +
          "<b>Critical path</b> to the worst part beneath it (" + esc(C.short[nodes[w].verdict]) + "), " + (crit.length - 1) + (crit.length === 2 ? " step: " : " steps: ") + chainHtml(crit) + "</p>";
      }
      if (chain.length > 1) {
        left += '<p class="vt-path"><svg viewBox="0 0 30 8" width="30" height="8" aria-hidden="true"><path d="M2,4H28" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-dasharray="1 4"/></svg>' +
          "<b>Longest chain</b> to the ground, " + (chain.length - 1) + " steps: " + chainHtml(chain) + "</p>";
      }
      var right = "<h5>What would fix it</h5>";
      if (L.fix.length) {
        right += '<p class="vt-muted vt-path">The parts that are not verified, the ones that hold back most others first.</p><ul class="vt-fix">' + L.fix.slice(0, 6).map(fixItem).join("") + "</ul>";
        if (L.fix.length > 6) right += '<details class="vt-fold"><summary>Show ' + (L.fix.length - 6) + ' more</summary><ul class="vt-fix">' + L.fix.slice(6).map(fixItem).join("") + "</ul></details>";
      } else {
        right += '<p class="vt-fine">Nothing to fix: every part in this cone is verified.</p>';
      }
      var v = L.ver;
      right += "<h5>Fine as it is</h5><p class=\"vt-fine\"><b>" + v[0] + " verified part" + (v[0] === 1 ? "" : "s") + "</b> beneath" +
        (v[0] ? "<br><span>" + v[1].map(function (t) { return esc(t[0]) + " " + t[1]; }).join(", ") + "</span>" : "") + "</p>";
      body.live.innerHTML = '<div class="vt-lens-grid"><div>' + left + "</div><div>" + right + "</div></div>" +
        '<p class="vt-muted vt-path">Below the ground line, each cell reads cone / all: how many of the cone\'s parts, and how many in all, have a counted check against that class of root.</p>';
      body.clear.hidden = false;
      pick.value = pipe ? id : "";
    }

    // ---- selection
    function mark(id) {
      all(".vt-sel, .vt-up, .vt-down, .vt-in", function (el) { el.classList.remove("vt-sel", "vt-up", "vt-down", "vt-in"); });
      inCone = {}; hov = null;
      if (sky) sky.classList.remove("vt-dim");
      if (overlay) overlay.innerHTML = "";
      showFoundation(null);
      if (!id) return;
      var members = coneIds(id);
      members.forEach(function (m) { inCone[m] = true; });
      var chip = ".vt-map [data-node], .vt-step[data-node]";
      if (nodes[id]) {
        all(chip, function (el) { if (el.getAttribute("data-node") === id) el.classList.add("vt-sel"); });
        nodes[id].depends.forEach(function (d) { all(".vt-map " + sel(d) + ", .vt-step" + sel(d), function (el) { el.classList.add("vt-down"); }); });
        nodes[id].dependents.forEach(function (d) { all(".vt-map " + sel(d) + ", .vt-step" + sel(d), function (el) { el.classList.add("vt-up"); }); });
        if (map) map.classList.add("vt-dim");
      } else if (map) map.classList.remove("vt-dim");
      if (sky) {
        sky.classList.add("vt-dim");
        members.forEach(function (m) { all(".vt-sky " + sel(m), function (el) { el.classList.add("vt-in"); }); });
        if (nodes[id]) all(".vt-sky " + sel(id), function (el) { el.classList.add("vt-sel"); });
      }
      drawLens(id);
      showFoundation(members);
    }
    function centreOnSelection(id) {   // the figure scrolls sideways on a small screen: bring the top of the cone into view
      var scroller = sky && sky.querySelector(".vt-sky-scroll"), t = pos[coneIds(id)[0]];
      if (!scroller || !svg || !t || scroller.scrollWidth <= scroller.clientWidth) return;
      var k = scroller.scrollWidth / svg.viewBox.baseVal.width;
      scroller.scrollLeft = Math.max(0, t.cx * k - scroller.clientWidth / 2);
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
      centreOnSelection(id);
      if (o.hash !== false) {
        try { history.replaceState(null, "", "#" + encodeURIComponent(id)); } catch (err) { /* a file: page cannot */ }
      }
      if (o.scroll && sky) sky.scrollIntoView({block: o.block || "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"});
      return true;
    }
    function clear() {
      if (current && decodeURIComponent(location.hash.slice(1)) === current) {
        try { history.replaceState(null, "", location.pathname + location.search); } catch (err) { /* ignore */ }
      }
      current = null; mark(null);
      if (cone) {
        body.live.innerHTML = '<p class="vt-muted vt-cone-hint">Select a square in the skyline, a part in the list below or a pipeline here. The skyline lights everything it relies on, down to the ground.</p>';
        body.clear.hidden = true; pick.value = "";
      }
    }
    function fromHash(first) {
      var h; try { h = decodeURIComponent(location.hash.slice(1)); } catch (err) { return; }
      if (!h || h === current || !C.cones[h]) return;
      select(h, {hash: false, scroll: first, block: "start"});
      // on load the page is still settling (images above the figure are lazy): scroll once it has
      if (first && sky) {
        var go = function () { if (current === h) sky.scrollIntoView({block: "start"}); };
        if (document.readyState === "complete") setTimeout(go, 50); else window.addEventListener("load", function () { setTimeout(go, 50); }, {once: true});
      }
    }

    // ---- wiring
    buildPanel();
    all(".vt [data-node]", function (el) {
      var inSky = !!el.closest(".vt-sky"), inMap = !!el.closest(".vt-map");
      if (inSky && el.tagName.toLowerCase() === "g") { el.setAttribute("tabindex", "0"); el.setAttribute("role", "button"); }
      el.addEventListener("click", function (ev) {
        if (ev.button || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;   // open in a new tab, as a link does
        if (inSky) ev.preventDefault();
        select(el.getAttribute("data-node"), inMap ? {fold: true} : {});
      });
      el.addEventListener("keydown", function (ev) {
        if ((ev.key === "Enter" || ev.key === " ") && el.tagName.toLowerCase() === "g") { ev.preventDefault(); el.dispatchEvent(new MouseEvent("click", {bubbles: true})); }
      });
      if (inSky) {
        var id = el.getAttribute("data-node");
        var over = function () { if (el.classList.contains("vt-in")) showHover(id); };
        el.addEventListener("mouseenter", over);
        el.addEventListener("focus", over);
        el.addEventListener("mouseleave", function () { showHover(null); });
        el.addEventListener("blur", function () { showHover(null); });
      }
    });
    panel.addEventListener("click", function (ev) {
      if (ev.target.classList && ev.target.classList.contains("vt-cone-jump") && sky) sky.scrollIntoView({block: "start"});
    });
    if (cone) {
      // a link in the panel (a path or a fix) selects that part; the hash change does the rest
      cone.addEventListener("click", function (ev) {
        var a = ev.target.closest && ev.target.closest("a[href^='#']");
        if (!a || ev.button || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
        var id = decodeURIComponent(a.getAttribute("href").slice(1));
        if (!C.cones[id]) return;
        ev.preventDefault();
        select(id, {});
      });
    }
    if (onHash) window.removeEventListener("hashchange", onHash);
    onHash = function () { fromHash(false); };
    window.addEventListener("hashchange", onHash);
    fromHash(true);
  }
  if (typeof document$ !== "undefined") document$.subscribe(init); else document.addEventListener("DOMContentLoaded", init);
})();
