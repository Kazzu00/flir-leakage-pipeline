/* FLIR adapter for cpietsch/vikus-viewer; see docs/vikus_explorer.md.
 * Display-only: no fitted geometry, memberships or source annotations change. */
(function () {
  "use strict";
  const originalClean = utils.clean;
  utils.clean = function (rows, config) {
    originalClean(rows, config);
    rows.forEach(function (row) {
      row._class_presence = JSON.parse(row._class_presence);
      row._split_memberships = JSON.parse(row._split_memberships);
    });
    document.getElementById("flir-experiment").textContent = config.flir.experiment_label;
    function showCount() {
      const active = rows.filter(function (row) { return row.active; }).length;
      document.getElementById("flir-status").textContent = active + " / " + rows.length + " contenidos · sólo lectura";
    }
    document.getElementById("flir-reset").onclick = function () {
      tags.reset();
      search.reset();
      canvas.resetZoom();
      showCount();
    };
    document.getElementById("flir-focus").onclick = function () {
      canvas.setView(rows.filter(function (row) { return row.active; }).map(function (row) { return row.id; }));
    };
    setInterval(showCount, 400);
    showCount();
  };
  if (!Modernizr.webgl || utils.isMobile()) {
    document.getElementById("flir-status").textContent = "VIKUS requiere WebGL y una ventana de escritorio de al menos 500 px.";
  }
})();
