const PERF_EVAL_SELECTION = (() => {
  "use strict";

  function resetSelection(checkboxes) {
    for (const checkbox of checkboxes) checkbox.checked = false;
  }

  function fillSelection(checkboxes, limit) {
    const boxes = Array.from(checkboxes);
    let selectedCount = boxes.filter((checkbox) => checkbox.checked).length;
    for (const checkbox of boxes) {
      if (selectedCount >= limit) break;
      if (checkbox.checked) continue;
      checkbox.checked = true;
      selectedCount += 1;
    }
  }

  function selectionSnapshot(checkboxes, limit, statusLocked, protocolReady) {
    const boxes = Array.from(checkboxes);
    const selected = boxes.filter((checkbox) => checkbox.checked);
    return {
      selectedCount: selected.length,
      selectedValues: selected.map((checkbox) => checkbox.value),
      countText: `선택 ${selected.length} / 최대 ${limit}장`,
      etaText: `≈ ${Math.round(selected.length * 2.2)}초`,
      executeDisabled: statusLocked || !protocolReady || selected.length === 0,
      helperDisabled: statusLocked,
      checkboxDisabled: boxes.map((checkbox) =>
        statusLocked || (selected.length >= limit && !checkbox.checked)),
    };
  }

  function paginateItems(items, page, pageSize) {
    const safeItems = Array.isArray(items) ? items : [];
    const safePageSize = Number.isInteger(pageSize) && pageSize > 0 ? pageSize : 1;
    const pageCount = Math.max(1, Math.ceil(safeItems.length / safePageSize));
    const requestedPage = Number.isInteger(page) ? page : 0;
    const safePage = Math.max(0, Math.min(requestedPage, pageCount - 1));
    const offset = safePage * safePageSize;
    const pageItems = safeItems.slice(offset, offset + safePageSize);
    return {
      items: pageItems,
      page: safePage,
      pageCount,
      start: pageItems.length ? offset + 1 : 0,
      end: offset + pageItems.length,
      total: safeItems.length,
    };
  }

  function statusFailure(status, previousState, previousRunId) {
    const state = typeof status?.state === "string" ? status.state : "unknown";
    const runId = typeof status?.run_id === "string" && status.run_id
      ? status.run_id : null;
    if (state !== "error" || (previousState === "error" && previousRunId === runId)) {
      return null;
    }
    const logs = Array.isArray(status.log)
      ? status.log.filter((line) => typeof line === "string" && line) : [];
    const cause = logs.length ? ` · ${logs[logs.length - 1]}` : "";
    return {
      runId,
      message: `실행 실패 · run_id ${runId || "알 수 없음"}${cause}`,
    };
  }

  function selectionMatches(left, right) {
    const leftValues = Array.isArray(left) ? [...left].sort() : [];
    const rightValues = Array.isArray(right) ? [...right].sort() : [];
    return leftValues.length === rightValues.length &&
      leftValues.every((value, index) => value === rightValues[index]);
  }

  function resultMatchesSelection(result, selectedValues, activeRunId) {
    return result?.run_id === activeRunId && Array.isArray(result.items) &&
      selectionMatches(
        selectedValues,
        result.items.map((item) => item?.item_id),
      );
  }

  return {
    resetSelection, fillSelection, selectionSnapshot, paginateItems, statusFailure,
    selectionMatches, resultMatchesSelection,
  };
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = PERF_EVAL_SELECTION;
}

if (typeof window !== "undefined") {
(() => {
  "use strict";

  const config = window.PERF_EVAL_CONFIG || {};
  const TERMINAL_STATES = new Set(["idle", "done", "error"]);
  const DISPLAY_RANGES = {psnr: [15, 35], ssim: [0.70, 1.00]};
  const MAX_TRY_ITEMS = 100;
  const TRY_PAGE_SIZE = 10;
  const tryButton = document.querySelector("#try-button");
  const selectionFillButton = document.querySelector("#selection-fill");
  const selectionResetButton = document.querySelector("#selection-reset");
  const tryPagePrevious = document.querySelector("#try-page-prev");
  const tryPageNext = document.querySelector("#try-page-next");
  const tryProgress = document.querySelector("#try-progress");
  const tryProgressTrack = document.querySelector("#try-progress-track");
  const inlineAlert = document.querySelector("#run-alert");
  let activeTryRunId = null;
  let activeTryTotal = 0;
  let previousState = "unknown";
  let previousStatusRunId = null;
  let statusLocked = true;
  let tryProtocolReady = false;
  let activeTryResult = null;
  let tryResultPage = 0;

  selectionFillButton.textContent = "모두 선택";

  if (!Number.isInteger(config.backendPort)) {
    showAlert("BACKEND_PORT 설정을 읽지 못했다.");
    setLocked(true);
    return;
  }

  const apiBase = `${window.location.protocol}//${window.location.hostname}:${config.backendPort}`;

  function apiUrl(path, parameters = {}) {
    const url = new URL(path, apiBase);
    for (const [key, value] of Object.entries(parameters)) url.searchParams.set(key, value);
    return url;
  }

  function selectionCheckboxes() {
    return [...document.querySelectorAll("#view-pair .item input")];
  }

  function selectedItemIds() {
    return selectionCheckboxes().filter((input) => input.checked).map((input) => input.value);
  }

  function updateSelection() {
    const checkboxes = selectionCheckboxes();
    const view = PERF_EVAL_SELECTION.selectionSnapshot(
      checkboxes, MAX_TRY_ITEMS, statusLocked, tryProtocolReady,
    );
    document.querySelector("#selection-count").textContent = view.countText;
    tryButton.textContent = statusLocked ? "실행 중…" : "생성";
    checkboxes.forEach((checkbox, index) => {
      checkbox.disabled = view.checkboxDisabled[index];
      checkbox.closest(".item").classList.toggle("is-checked", checkbox.checked);
    });
    tryButton.disabled = view.executeDisabled;
    selectionFillButton.disabled = view.helperDisabled;
    selectionResetButton.disabled = view.helperDisabled;
    if (activeTryResult && !PERF_EVAL_SELECTION.selectionMatches(
      view.selectedValues,
      activeTryResult.items.map((item) => item.item_id),
    )) {
      activeTryResult = null;
      document.querySelector("#try-result").hidden = true;
      clearSelectedResults("현재 선택으로 측정 실행 필요");
    }
  }

  function setLocked(locked) {
    statusLocked = locked;
    updateSelection();
  }

  function showAlert(message) {
    inlineAlert.textContent = message;
    inlineAlert.hidden = !message;
  }

  function renderTryProgress(status) {
    const view = PERF_RUN_PROGRESS.snapshot(status, activeTryTotal);
    tryProgress.hidden = false;
    tryProgress.dataset.state = view.state;
    document.querySelector("#try-progress-stage").textContent = view.label;
    document.querySelector("#try-progress-count").textContent = view.total
      ? `${view.done}/${view.total}` : `${view.done}/—`;
    document.querySelector("#try-progress-percent").textContent = `${view.percent}%`;
    document.querySelector("#try-progress-fill").style.setProperty(
      "--done", `${view.percent}%`,
    );
    tryProgressTrack.setAttribute("aria-valuenow", String(view.percent));
    tryProgressTrack.setAttribute("aria-valuetext", `${view.label} ${view.percent}%`);
  }

  function clampPercent(value, low, high) {
    return Math.max(0, Math.min(100, ((value - low) / (high - low)) * 100));
  }

  function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  async function loadTryConfig() {
    tryProtocolReady = false;
    updateSelection();
    try {
      const response = await fetch(apiUrl("/api/try/config"));
      if (!response.ok) throw new Error(`try config ${response.status}`);
      const payload = await response.json();
      if (!isFiniteNumber(payload.strength)) return;
      tryProtocolReady = true;
    } catch (error) {
      showAlert(`측정 설정 조회 실패: ${error.message}`);
    } finally {
      updateSelection();
    }
  }

  function applyStatus(status) {
    const state = typeof status.state === "string" ? status.state : "unknown";
    const statusRunId = typeof status.run_id === "string" && status.run_id
      ? status.run_id : null;
    const isTryStatus = Boolean(statusRunId && statusRunId.startsWith("try-"));
    const failure = PERF_EVAL_SELECTION.statusFailure(
      status, previousState, previousStatusRunId,
    );
    if (failure) showAlert(failure.message);
    if (isTryStatus) {
      activeTryRunId = statusRunId;
      if (Number.isFinite(status.total) && status.total > 0) {
        activeTryTotal = status.total;
      }
      renderTryProgress(status);
    }

    const locked = !TERMINAL_STATES.has(state);
    setLocked(locked);

    const newlyDone = state === "done" &&
      (previousState !== "done" || previousStatusRunId !== statusRunId);
    if (newlyDone && statusRunId) {
      if (isTryStatus) loadTryResults(statusRunId);
    }
    previousState = state;
    previousStatusRunId = statusRunId;
  }

  async function loadStatus() {
    try {
      const response = await fetch(apiUrl("/api/status"));
      if (!response.ok) throw new Error(`status ${response.status}`);
      applyStatus(await response.json());
    } catch (error) {
      setLocked(true);
      showAlert(`상태 조회 실패: ${error.message}`);
    }
  }

  function renderDataset(payload) {
    const quotaStrip = document.querySelector("#quota-strip");
    const tree = document.querySelector("#tree-groups");
    quotaStrip.replaceChildren();
    tree.replaceChildren();
    const itemsByGroup = new Map();
    for (const item of payload.items) {
      if (!itemsByGroup.has(item.group)) itemsByGroup.set(item.group, []);
      itemsByGroup.get(item.group).push(item);
    }
    for (const group of payload.groups) {
      const cell = document.createElement("span");
      cell.className = "quota-cell";
      const name = document.createElement("b");
      name.textContent = group.group;
      cell.append(name, document.createTextNode(String(group.count)));
      quotaStrip.append(cell);

      const details = document.createElement("details");
      details.className = "group";
      const summary = document.createElement("summary");
      summary.className = "group-head";
      const groupName = document.createElement("span");
      groupName.className = "group-name";
      groupName.textContent = group.group;
      const count = document.createElement("span");
      count.className = "group-count";
      count.textContent = `${group.count}장`;
      summary.append(groupName, count);
      const list = document.createElement("div");
      list.className = "item-list";
      for (const item of itemsByGroup.get(group.group) || []) {
        const label = document.createElement("label");
        label.className = "item";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.value = item.item_id;
        checkbox.addEventListener("change", updateSelection);
        const type = document.createElement("span");
        type.className = "item-type";
        type.textContent = item.product_type;
        const image = document.createElement("span");
        image.className = "item-image";
        image.textContent = item.image_id;
        label.append(checkbox, type, image);
        list.append(label);
      }
      details.append(summary, list);
      tree.append(details);
    }
    const first = document.querySelector("#view-pair .item input");
    if (first) first.checked = true;
    updateSelection();
  }

  async function loadDataset() {
    try {
      const [datasetResponse, itemsResponse] = await Promise.all([
        fetch(apiUrl("/api/dataset")),
        fetch(apiUrl("/api/dataset/items")),
      ]);
      if (!datasetResponse.ok) throw new Error(`dataset ${datasetResponse.status}`);
      if (!itemsResponse.ok) throw new Error(`dataset items ${itemsResponse.status}`);
      renderDataset(await itemsResponse.json());
    } catch (error) {
      showAlert(`데이터셋 조회 실패: ${error.message}`);
    }
  }

  function setMetric(name, value, target, options) {
    const card = document.querySelector(`#metric-${name}`);
    const valueElement = document.querySelector(`#${name}-value`);
    const targetElement = document.querySelector(`#${name}-target`);
    const badge = document.querySelector(`#${name}-badge`);
    const tolerance = document.querySelector(`#${name}-tolerance`);
    const targetSymbol = options.direction === "le" ? "≤" : "≥";
    targetElement.textContent = isFiniteNumber(target)
      ? `목표 ${targetSymbol} ${target} · ${options.description}`
      : `목표 — · ${options.description}`;
    if (!isFiniteNumber(value) || !isFiniteNumber(target)) {
      valueElement.textContent = "—";
      badge.hidden = true;
      tolerance.hidden = true;
      delete card.dataset.verdict;
      return;
    }
    const passed = options.direction === "le" ? value <= target : value >= target;
    const verdict = passed ? "pass" : "fail";
    card.dataset.verdict = verdict;
    badge.dataset.verdict = verdict;
    badge.textContent = passed ? "PASS" : "FAIL";
    badge.hidden = false;
    tolerance.hidden = false;
    valueElement.textContent = value.toFixed(options.decimals);
    if (options.unit) {
      const unit = document.createElement("span");
      unit.className = "unit";
      unit.textContent = options.unit;
      valueElement.append(unit);
    }
    const [low, high] = DISPLAY_RANGES[name];
    tolerance.style.setProperty("--tick", `${clampPercent(target, low, high)}%`);
    tolerance.style.setProperty("--pos", `${clampPercent(value, low, high)}%`);
  }

  function normalizePerImage(values) {
    if (!Array.isArray(values)) return [];
    return values.map((entry) => typeof entry === "object" && entry !== null
      ? entry.value : entry).filter(isFiniteNumber);
  }

  function setHistogram(name, rawValues, target) {
    const values = normalizePerImage(rawValues);
    const plot = document.querySelector(`#${name}-histogram`);
    const bars = plot.querySelector(".dist-bars");
    const count = document.querySelector(`#${name}-dist-n`);
    const foot = document.querySelector(`#${name}-dist-foot`);
    bars.replaceChildren();
    count.textContent = `n=${values.length || "—"}`;
    if (!values.length || !isFiniteNumber(target)) {
      foot.textContent = "장별 값 없음";
      plot.querySelector(".dist-tick").hidden = true;
      return;
    }
    const [low, high] = DISPLAY_RANGES[name];
    const bins = Array(16).fill(0);
    for (const value of values) {
      const normalized = Math.max(0, Math.min(0.999999, (value - low) / (high - low)));
      bins[Math.floor(normalized * bins.length)] += 1;
    }
    const maximum = Math.max(...bins, 1);
    for (const bin of bins) {
      const bar = document.createElement("i");
      bar.style.setProperty("--h", `${(bin / maximum) * 100}%`);
      bars.append(bar);
    }
    plot.style.setProperty("--tick", `${clampPercent(target, low, high)}%`);
    plot.querySelector(".dist-tick").hidden = false;
    foot.textContent = `장별 실측값 분포 · 목표선 ${target}`;
  }

  function clearSelectedResults(message) {
    document.querySelector("#result-empty").hidden = false;
    document.querySelector("#result-empty").textContent = message;
    setMetric("psnr", null, null, {direction: "ge", decimals: 2, unit: "dB", description: "높을수록 좋음"});
    setMetric("ssim", null, null, {direction: "ge", decimals: 3, description: "높을수록 좋음"});
    document.querySelector("#psnr-foot").textContent = "—";
    document.querySelector("#ssim-foot").textContent = "—";
    setHistogram("psnr", [], null);
    setHistogram("ssim", [], null);
  }

  function renderSelectedResults(result) {
    const items = Array.isArray(result.items) ? result.items : [];
    const count = items.length;
    document.querySelector("#result-empty").hidden = true;
    setMetric("psnr", result.metrics?.psnr?.mean, result.targets?.psnr, {
      direction: "ge", decimals: 2, unit: "dB", description: "높을수록 좋음",
    });
    setMetric("ssim", result.metrics?.ssim?.mean, result.targets?.ssim, {
      direction: "ge", decimals: 3, description: "높을수록 좋음",
    });
    document.querySelector("#psnr-foot").textContent = `선택 데이터 평균 · n=${count}`;
    document.querySelector("#ssim-foot").textContent = `선택 데이터 평균 · n=${count}`;
    setHistogram("psnr", items.map((item) => item.psnr), result.targets?.psnr);
    setHistogram("ssim", items.map((item) => item.ssim), result.targets?.ssim);
  }

  function imageCell(path, label) {
    const cell = document.createElement("div");
    cell.className = "compare-cell";
    const image = document.createElement("img");
    image.src = apiUrl("/api/image", {path});
    image.alt = label;
    const tag = document.createElement("span");
    tag.className = "compare-tag";
    tag.textContent = label;
    cell.append(image, tag);
    return cell;
  }

  function tryMetric(name, value, target, decimals, unit = "") {
    const metric = document.createElement("div");
    metric.className = "tm";
    const label = document.createElement("span");
    label.className = "tm-name";
    label.textContent = name;
    const shown = document.createElement("span");
    shown.className = "tm-value";
    shown.textContent = isFiniteNumber(value) ? value.toFixed(decimals) + unit : "—";
    const badge = document.createElement("span");
    badge.className = "badge";
    if (isFiniteNumber(value) && isFiniteNumber(target)) {
      const passed = value >= target;
      badge.dataset.verdict = passed ? "pass" : "fail";
      badge.textContent = passed ? "PASS" : "FAIL";
    } else {
      badge.hidden = true;
    }
    metric.append(label, shown, badge);
    return metric;
  }

  function renderTryItem(item, targets) {
    const row = document.createElement("article");
    row.className = "try-row";
    const images = document.createElement("div");
    images.className = "try-images";
    images.append(imageCell(item.input_path, "원본"), imageCell(item.output_path, "생성물"));
    const side = document.createElement("div");
    side.className = "try-side";
    const label = document.createElement("p");
    label.className = "try-label";
    const type = document.createElement("span");
    type.className = "item-type";
    type.textContent = item.product_type;
    const imageId = document.createElement("span");
    imageId.className = "item-image";
    imageId.textContent = item.image_id;
    label.append(type, document.createTextNode(" "), imageId);
    const meta = document.createElement("p");
    meta.className = "try-meta";
    meta.textContent = `${item.group} · item_id ${item.item_id}`;
    const metrics = document.createElement("div");
    metrics.className = "try-metrics";
    metrics.append(
      tryMetric("PSNR", item.psnr, targets.psnr, 2, " dB"),
      tryMetric("SSIM", item.ssim, targets.ssim, 3),
    );
    side.append(label, meta, metrics);
    row.append(images, side);
    return row;
  }

  function renderTryResultPage() {
    if (!activeTryResult) return;
    const view = PERF_EVAL_SELECTION.paginateItems(
      activeTryResult.items, tryResultPage, TRY_PAGE_SIZE,
    );
    tryResultPage = view.page;
    const list = document.querySelector("#try-result-list");
    list.replaceChildren(...view.items.map((item) =>
      renderTryItem(item, activeTryResult.targets)));
    document.querySelector("#try-page-status").textContent =
      `${view.start}–${view.end} / ${view.total} · ${view.page + 1} / ${view.pageCount} 페이지`;
    tryPagePrevious.disabled = view.page === 0;
    tryPageNext.disabled = view.page >= view.pageCount - 1;
    document.querySelector("#try-pagination").hidden = view.pageCount <= 1;
  }

  async function loadTryResults(runId) {
    try {
      const response = await fetch(apiUrl("/api/try/results", {run_id: runId}));
      if (!response.ok) throw new Error(`try results ${response.status}`);
      const result = await response.json();
      if (!PERF_EVAL_SELECTION.resultMatchesSelection(
        result, selectedItemIds(), activeTryRunId,
      )) {
        clearSelectedResults("현재 선택으로 측정 실행 필요");
        return;
      }
      const section = document.querySelector("#try-result");
      renderSelectedResults(result);
      activeTryResult = result;
      tryResultPage = 0;
      renderTryResultPage();
      section.hidden = false;
    } catch (error) {
      showAlert(`시험 결과 조회 실패: ${error.message}`);
    }
  }

  async function startTry() {
    showAlert("");
    const itemIds = selectedItemIds();
    if (!itemIds.length || itemIds.length > MAX_TRY_ITEMS) {
      showAlert(`시험할 장을 1장 이상 ${MAX_TRY_ITEMS}장 이하로 선택해야 한다.`);
      return;
    }
    clearSelectedResults("선택 데이터 측정 중");
    document.querySelector("#try-result").hidden = true;
    activeTryResult = null;
    activeTryTotal = itemIds.length;
    renderTryProgress({state: "running", done: 0, total: activeTryTotal});
    setLocked(true);
    try {
      const response = await fetch(apiUrl("/api/try"), {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({item_ids: itemIds}),
      });
      const payload = await response.json();
      if (response.status === 409) {
        tryProgress.hidden = true;
        showAlert(`이미 실행 중이다 · run_id ${payload.run_id}`);
        return;
      }
      if (response.status === 503) await loadTryConfig();
      if (!response.ok) throw new Error(payload.detail || `try ${response.status}`);
      activeTryRunId = payload.run_id;
      previousState = "running";
      applyStatus({
        state: "running", done: 0, total: activeTryTotal,
        log: [], run_id: payload.run_id,
      });
    } catch (error) {
      tryProgress.hidden = true;
      showAlert(`시험 요청 실패: ${error.message}`);
      setLocked(false);
    }
  }

  selectionFillButton.addEventListener("click", () => {
    const checkboxes = selectionCheckboxes();
    PERF_EVAL_SELECTION.fillSelection(checkboxes, MAX_TRY_ITEMS);
    updateSelection();
  });
  selectionResetButton.addEventListener("click", () => {
    const checkboxes = selectionCheckboxes();
    PERF_EVAL_SELECTION.resetSelection(checkboxes);
    updateSelection();
  });
  tryPagePrevious.addEventListener("click", () => {
    tryResultPage -= 1;
    renderTryResultPage();
  });
  tryPageNext.addEventListener("click", () => {
    tryResultPage += 1;
    renderTryResultPage();
  });
  tryButton.addEventListener("click", startTry);
  loadDataset();
  loadTryConfig();
  clearSelectedResults("선택 실행 대기");
  loadStatus();
  window.setInterval(loadStatus, 1000);
})();
}
