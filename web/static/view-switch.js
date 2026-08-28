const PERF_VIEW_SWITCH = (() => {
  "use strict";

  const VALID_VIEWS = new Set(["pair", "fid", "fidv2"]);

  function resolveView(pathname, search) {
    if (pathname.endsWith("/fid.html")) return "fid";
    const searchParams = new URLSearchParams(search);
    const requested = searchParams.get("view");
    return VALID_VIEWS.has(requested) ? requested : "pair";
  }

  return {resolveView};
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = PERF_VIEW_SWITCH;
}

if (typeof window !== "undefined") {
  (() => {
    "use strict";

    const buttons = [...document.querySelectorAll("[data-metric-view]")];
    const panels = [...document.querySelectorAll("[data-metric-panel]")];

    function activateView(view) {
      for (const button of buttons) {
        const selected = button.dataset.metricView === view;
        button.setAttribute("aria-selected", String(selected));
      }
      for (const panel of panels) {
        panel.hidden = panel.dataset.metricPanel !== view;
      }
    }

    function viewUrl(view) {
      const url = new URL(window.location.href);
      url.pathname = "/";
      if (view === "pair") url.searchParams.delete("view");
      else url.searchParams.set("view", view);
      return `${url.pathname}${url.search}${url.hash}`;
    }

    for (const button of buttons) {
      button.addEventListener("click", () => {
        const view = button.dataset.metricView;
        if (!view) return;
        activateView(view);
        window.history.pushState({view}, "", viewUrl(view));
      });
    }

    window.addEventListener("popstate", () => {
      activateView(PERF_VIEW_SWITCH.resolveView(window.location.pathname, window.location.search));
    });

    activateView(PERF_VIEW_SWITCH.resolveView(window.location.pathname, window.location.search));
  })();
}
