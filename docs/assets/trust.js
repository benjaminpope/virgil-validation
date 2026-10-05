// The Trust page: select a part to see how it was checked, and what it
// relies on (scripts/trust.py embeds the data as #vt-data).
(function () {
  function init() {
    var holder = document.getElementById("vt-data");
    var panel = document.getElementById("vt-detail");
    if (!holder || !panel) return;
    var data = JSON.parse(holder.textContent);
    var nodes = data.nodes, verdicts = data.verdicts, roots = data.roots;
    var map = document.querySelector(".vt-map");
    function esc(s) { var d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; }
    function fmt(v) {
      if (v == null || typeof v !== "number") return esc(v);
      return (v !== 0 && (Math.abs(v) < 1e-2 || Math.abs(v) >= 1e4)) ? v.toExponential(1) : String(Number(v.toPrecision(3)));
    }
    function select(id) {
      var n = nodes[id];
      if (!n) return;
      document.querySelectorAll(".vt-sel, .vt-up, .vt-down").forEach(function (el) { el.classList.remove("vt-sel", "vt-up", "vt-down"); });
      document.querySelectorAll('[data-node="' + CSS.escape(id) + '"]').forEach(function (el) { el.classList.add("vt-sel"); });
      n.depends.forEach(function (d) { document.querySelectorAll('[data-node="' + CSS.escape(d) + '"]').forEach(function (el) { el.classList.add("vt-down"); }); });
      n.dependents.forEach(function (d) { document.querySelectorAll('[data-node="' + CSS.escape(d) + '"]').forEach(function (el) { el.classList.add("vt-up"); }); });
      if (map) map.classList.add("vt-dim");
      var v = verdicts[n.verdict];
      var html = "<h4>" + esc(n.id) + "</h4><div><strong>" + esc(v[0]) + "</strong>: " + esc(v[1]) + ".</div>";
      if (n.findings.length) html += "<div>Open: " + n.findings.map(function (f) { return '<a href="#' + esc(f) + '">' + esc(f) + "</a>"; }).join(", ") + "</div>";
      if (n.because.length) html += "<div>" + (n.verdict.indexOf("relies") === 0 ? "Relies on " : "Also relies on ") + n.because.map(function (b) {
        return "<code>" + esc(b.id) + "</code>" + (b.findings.length ? " (" + b.findings.map(function (f) { return '<a href="#' + esc(f) + '">' + esc(f) + "</a>"; }).join(", ") + ")" : "");
      }).join(", ") + (n.verdict.indexOf("relies") === 0 ? "; it counts as verified once that is fixed." : "; fixing that alone would not verify it.") + "</div>";
      var rs = n.strong.concat(n.weak);
      html += "<div>Checked against: " + (rs.length ? rs.map(function (r) { return '<span class="vt-root">' + esc(roots[r] || r) + "</span>"; }).join("") : "nothing yet") + "</div>";
      if (n.changed && n.changed.length) html += '<div class="vt-muted">virgil has changed ' + n.changed.map(function (f) { return "<code>" + esc(f) + "</code>"; }).join(", ") + " since this evidence.</div>";
      var checks = n.checks.filter(function (c) { return c.kind !== "guard" && c.outcome !== "skipped"; });
      if (checks.length) {
        html += '<ul class="vt-checks">' + checks.slice(0, 12).map(function (c) {
          var out = c.outcome === "passed" ? "" : " <em>(" + esc(c.outcome) + ")</em>";
          var val = c.headline ? " — " + esc(c.headline) + " = " + fmt(c.value) : "";
          var name = c.url ? '<a href="' + esc(c.url) + '">' + esc(c.name) + "</a>" : esc(c.name);
          return "<li>" + name + out + val + (c.doc ? '<div class="vt-muted">' + esc(c.doc) + "</div>" : "") + "</li>";
        }).join("") + "</ul>";
        if (checks.length > 12) html += '<div class="vt-muted">and ' + (checks.length - 12) + " more in the table below.</div>";
      }
      panel.innerHTML = html;
    }
    document.querySelectorAll(".vt [data-node]").forEach(function (el) {
      el.addEventListener("click", function () { select(el.getAttribute("data-node")); });
    });
  }
  if (typeof document$ !== "undefined") document$.subscribe(init); else document.addEventListener("DOMContentLoaded", init);
})();
