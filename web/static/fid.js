const PERF_FID_VIEW = (() => {
  "use strict";

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

  function selectionMatches(left, right) {
    const leftValues = Array.isArray(left) ? [...left].sort() : [];
    const rightValues = Array.isArray(right) ? [...right].sort() : [];
    return leftValues.length === rightValues.length &&
      leftValues.every((value, index) => value === rightValues[index]);
  }

  return {paginateItems, selectionMatches};
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = PERF_FID_VIEW;
}

if (typeof window !== "undefined") {
(() => {
  "use strict";

  const config = window.PERF_EVAL_CONFIG || {};
  const TERMINAL_STATES = new Set(["idle", "done", "error"]);
  const MIN_FID_ITEMS = 2;
  const MAX_FID_ITEMS = 500;
  const FID_DISPLAY_RANGE = [0, 20];
  const FID_PAGE_SIZE = 10;
  const runButton = document.querySelector("#fid-run-button");
  const selectionFillButton = document.querySelector("#fid-selection-fill");
  const selectionResetButton = document.querySelector("#fid-selection-reset");
  const alertBox = document.querySelector("#fid-run-alert");
  const fidProgress = document.querySelector("#fid-progress");
  const fidProgressTrack = document.querySelector("#fid-progress-track");
  let protocolReady = false;
  let statusReady = false;
  let runLocked = true;
  let activeFidRunId = null;
  let activeFidTotal = 0;
  let activeFidResult = null;
  let previousState = "unknown";
  let previousStatusRunId = null;
  let fidImageItems = [];
  let fidImagePage = 0;

  function setText(selector, value) {
    document.querySelector(selector).textContent = String(value);
  }

  function showAlert(message) {
    alertBox.textContent = message || "";
    alertBox.hidden = !message;
  }

  function selectionCheckboxes() {
    return [...document.querySelectorAll("#fid-tree-groups .item input")];
  }

  function selectedItemIds() {
    return selectionCheckboxes()
      .filter((checkbox) => checkbox.checked)
      .map((checkbox) => checkbox.value);
  }

  function showEmptyResult(message) {
    activeFidResult = null;
    fidImageItems = [];
    fidImagePage = 0;
    document.querySelector("#fid-result-empty").hidden = false;
    setText("#fid-result-empty", message);
    document.querySelector("#fid-result-body").hidden = true;
    setText("#fid-value", "—");
    setText("#fid-target", "목표 확인 중");
    setText("#fid-metric-foot", "선택 집합의 단일 거리 · 장별 평균 아님");
    const metric = document.querySelector("#fid-metric");
    const badge = document.querySelector("#fid-badge");
    delete metric.dataset.verdict;
    delete badge.dataset.verdict;
    badge.textContent = "—";
    badge.hidden = true;
    document.querySelector("#fid-tolerance").hidden = true;
    document.querySelector("#fid-scale").hidden = true;
    for (const selector of [
      "#fid-result-model",
      "#fid-result-strength",
      "#fid-result-count",
      "#fid-result-clean-fid",
      "#fid-result-seed",
      "#fid-result-determinism",
      "#fid-result-created-at",
      "#fid-result-manifest",
      "#fid-result-attribution",
    ]) setText(selector, "—");
    document.querySelector("#fid-image-list").replaceChildren();
    document.querySelector("#fid-page-buttons").replaceChildren();
    document.querySelector("#fid-images").hidden = true;
  }

  function updateSelection() {
    const checkboxes = selectionCheckboxes();
    const selectedIds = selectedItemIds();
    const count = selectedIds.length;
    setText("#fid-selection-count", `선택 ${count} / 최대 ${MAX_FID_ITEMS}장`);
    runButton.textContent = runLocked ? "실행 중…" : "측정";
    runButton.disabled = !protocolReady || !statusReady || runLocked ||
      count < MIN_FID_ITEMS || count > MAX_FID_ITEMS;
    checkboxes.forEach((checkbox) => {
      checkbox.disabled = runLocked || (count >= MAX_FID_ITEMS && !checkbox.checked);
      checkbox.closest(".item").classList.toggle("is-checked", checkbox.checked);
    });
    selectionFillButton.disabled = runLocked || checkboxes.length === 0;
    selectionResetButton.disabled = runLocked || checkboxes.length === 0;
    if (activeFidResult && !PERF_FID_VIEW.selectionMatches(
      selectedIds,
      activeFidResult.measurement?.item_ids,
    )) {
      showEmptyResult("현재 선택으로 측정 실행 필요");
    }
  }

  function setLocked(locked) {
    runLocked = locked;
    updateSelection();
  }

  function renderFidProgress(status) {
    const expectedTotal = activeFidTotal || selectedItemIds().length;
    const view = PERF_RUN_PROGRESS.snapshot(status, expectedTotal);
    fidProgress.hidden = false;
    fidProgress.dataset.state = view.state;
    setText("#fid-progress-stage", view.label);
    setText("#fid-progress-count", view.total ? `${view.done}/${view.total}` : `${view.done}/—`);
    setText("#fid-progress-percent", `${view.percent}%`);
    document.querySelector("#fid-progress-fill").style.setProperty(
      "--done", `${view.percent}%`,
    );
    fidProgressTrack.setAttribute("aria-valuenow", String(view.percent));
    fidProgressTrack.setAttribute("aria-valuetext", `${view.label} ${view.percent}%`);
  }

  function renderDatasetItems(payload) {
    const tree = document.querySelector("#fid-tree-groups");
    tree.replaceChildren();
    const itemsByGroup = new Map();
    for (const item of payload.items || []) {
      if (!itemsByGroup.has(item.group)) itemsByGroup.set(item.group, []);
      itemsByGroup.get(item.group).push(item);
    }
    for (const group of payload.groups || []) {
      const details = document.createElement("details");
      details.className = "group";
      const summary = document.createElement("summary");
      summary.className = "group-head";
      const name = document.createElement("span");
      name.className = "group-name";
      name.textContent = group.group;
      const count = document.createElement("span");
      count.className = "group-count";
      count.textContent = `${group.count}장`;
      summary.append(name, count);
      const list = document.createElement("div");
      list.className = "item-list";
      for (const item of itemsByGroup.get(group.group) || []) {
        const label = document.createElement("label");
        label.className = "item";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.value = item.item_id;
        checkbox.addEventListener("change", updateSelection);
        const product = document.createElement("span");
        product.className = "item-type";
        product.textContent = item.product_type;
        const imageId = document.createElement("span");
        imageId.className = "item-image";
        imageId.textContent = item.image_id;
        label.append(checkbox, product, imageId);
        list.append(label);
      }
      details.append(summary, list);
      tree.append(details);
    }
    selectionCheckboxes().slice(0, MIN_FID_ITEMS).forEach((checkbox) => {
      checkbox.checked = true;
    });
    updateSelection();
  }

  async function loadDataset() {
    const itemsResponse = await fetch(apiUrl("/api/fid/dataset/items"));
    if (!itemsResponse.ok) throw new Error(`FID500 items ${itemsResponse.status}`);
    renderDatasetItems(await itemsResponse.json());
  }

  async function loadProtocol() {
    const response = await fetch(apiUrl("/api/fid/config"));
    if (!response.ok) throw new Error(`FID config ${response.status}`);
    const payload = await response.json();
    const validStrength = typeof payload.strength === "number" &&
      Number.isFinite(payload.strength);
    protocolReady = payload.selected_available === true && validStrength;
    updateSelection();
  }

  function clampPercent(value, low, high) {
    return Math.max(0, Math.min(100, ((value - low) / (high - low)) * 100));
  }

  function jsonText(value) {
    return JSON.stringify(value, null, 2);
  }

  function renderResult(result) {
    const measurement = result.measurement;
    const protocol = result.protocol;
    const dataset = result.dataset;
    const selectedIds = selectedItemIds();
    const recordedIds = measurement?.item_ids;
    const fid = measurement?.fid;
    const target = measurement?.target?.value;
    const verdict = measurement?.verdict;
    if (
      result.run_id !== activeFidRunId ||
      measurement?.selection_mode !== "selected" ||
      !PERF_FID_VIEW.selectionMatches(selectedIds, recordedIds) ||
      measurement?.count !== recordedIds?.length ||
      typeof fid !== "number" || !Number.isFinite(fid) ||
      typeof target !== "number" || !Number.isFinite(target) ||
      !["PASS", "FAIL"].includes(verdict)
    ) {
      throw new Error("현재 선택과 FID 측정 결과가 일치하지 않는다.");
    }

    activeFidResult = result;
    document.querySelector("#fid-result-empty").hidden = true;
    document.querySelector("#fid-result-body").hidden = false;
    setText("#fid-value", fid.toFixed(2));
    setText("#fid-target", `목표 ${measurement.target.operator} ${target} · 낮을수록 좋음`);
    setText(
      "#fid-metric-foot",
      `선택 ${measurement.count}장 X·Y 집합의 단일 FID · 장별 평균 아님`,
    );

    const verdictName = verdict.toLowerCase();
    const metric = document.querySelector("#fid-metric");
    const badge = document.querySelector("#fid-badge");
    metric.dataset.verdict = verdictName;
    badge.dataset.verdict = verdictName;
    badge.textContent = verdict;
    badge.hidden = false;

    const [low, high] = FID_DISPLAY_RANGE;
    const tick = clampPercent(target, low, high);
    const position = clampPercent(fid, low, high);
    const tolerance = document.querySelector("#fid-tolerance");
    const scale = document.querySelector("#fid-scale");
    tolerance.style.setProperty("--tick", `${tick}%`);
    tolerance.style.setProperty("--pos", `${position}%`);
    scale.style.setProperty("--tick", `${tick}%`);
    tolerance.hidden = false;
    scale.hidden = false;
    setText("#fid-scale-low", low);
    setText("#fid-scale-target", target);
    setText("#fid-scale-high", high);

    setText("#fid-result-model", protocol.model);
    setText("#fid-result-strength", protocol.strength);
    setText("#fid-result-count", measurement.count);
    setText("#fid-result-clean-fid", jsonText(protocol.clean_fid));
    setText("#fid-result-seed", protocol.seed);
    setText("#fid-result-determinism", jsonText(protocol.determinism));
    setText("#fid-result-created-at", result.created_at);
    setText("#fid-result-manifest", dataset.manifest_sha256);
    setText(
      "#fid-result-attribution",
      [
        dataset.source_dataset.name,
        dataset.source_dataset.attribution,
        dataset.source_dataset.url,
      ].join(" · "),
    );
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

  function imageFilename(path) {
    const parts = String(path).split("/");
    return parts[parts.length - 1] || String(path);
  }

  function renderImageItem(item) {
    const row = document.createElement("article");
    row.className = "fid-image-row";
    const heading = document.createElement("p");
    heading.className = "fid-image-label";
    heading.textContent = `#${item.index} · ${imageFilename(item.input_path)}`;
    const pair = document.createElement("div");
    pair.className = "fid-image-pair";
    pair.append(
      imageCell(item.input_path, `${item.index}번 원본 x`),
      imageCell(item.output_path, `${item.index}번 생성물 y`),
    );
    row.append(heading, pair);
    return row;
  }

  function renderPageButtons(view) {
    const pages = document.querySelector("#fid-page-buttons");
    pages.replaceChildren();
    for (let index = 0; index < view.pageCount; index += 1) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "fid-page-number";
      button.textContent = String(index + 1);
      button.setAttribute("aria-label", `${index + 1}페이지`);
      if (index === view.page) {
        button.setAttribute("aria-current", "page");
        button.disabled = true;
      }
      button.addEventListener("click", () => {
        fidImagePage = index;
        renderImagePage();
        document.querySelector("#fid-images-title").scrollIntoView({block: "start"});
      });
      pages.append(button);
    }
  }

  function renderImagePage() {
    const view = PERF_FID_VIEW.paginateItems(fidImageItems, fidImagePage, FID_PAGE_SIZE);
    fidImagePage = view.page;
    document.querySelector("#fid-image-list").replaceChildren(
      ...view.items.map(renderImageItem),
    );
    setText(
      "#fid-image-page-status",
      `${view.start}–${view.end} / ${view.total} · ${view.page + 1} / ${view.pageCount} 페이지`,
    );
    renderPageButtons(view);
    document.querySelector("#fid-images").hidden = view.total === 0;
  }

  async function loadSelectedRun(runId) {
    const [resultResponse, imagesResponse] = await Promise.all([
      fetch(apiUrl("/api/fid/try/results", {run_id: runId})),
      fetch(apiUrl("/api/fid/try/images", {run_id: runId})),
    ]);
    if (!resultResponse.ok) throw new Error(`selected FID result ${resultResponse.status}`);
    if (!imagesResponse.ok) throw new Error(`selected FID images ${imagesResponse.status}`);
    const result = await resultResponse.json();
    const images = await imagesResponse.json();
    if (runId !== activeFidRunId || images.run_id !== runId) return;
    renderResult(result);
    const items = Array.isArray(images.items) ? images.items : [];
    if (items.length !== result.measurement.count) {
      throw new Error("선택 FID 결과와 이미지 수가 일치하지 않는다.");
    }
    fidImageItems = items;
    fidImagePage = 0;
    renderImagePage();
  }

  function applyStatus(status) {
    const state = typeof status.state === "string" ? status.state : "unknown";
    const statusRunId = typeof status.run_id === "string" && status.run_id
      ? status.run_id : null;
    const isActiveFidRun = Boolean(
      activeFidRunId && statusRunId === activeFidRunId &&
      statusRunId.startsWith("fidtry-"),
    );
    statusReady = true;
    if (isActiveFidRun) {
      if (Number.isFinite(status.total) && status.total > 0) activeFidTotal = status.total;
      renderFidProgress(status);
      const logs = Array.isArray(status.log) ? status.log : [];
      setText("#fid-log-tail", logs.length ? logs.join("\n") : "로그 대기 중");
      document.querySelector("#fid-log-tail").scrollTop =
        document.querySelector("#fid-log-tail").scrollHeight;
      if (state === "error" &&
          (previousState !== "error" || previousStatusRunId !== statusRunId)) {
        showAlert(logs.at(-1) || "선택 FID 측정이 실패했다.");
      }
    }

    setLocked(!TERMINAL_STATES.has(state));
    const newlyDone = isActiveFidRun && state === "done" &&
      (previousState !== "done" || previousStatusRunId !== statusRunId);
    if (newlyDone) {
      loadSelectedRun(statusRunId).catch((error) => {
        showEmptyResult("현재 선택 결과 조회 실패");
        showAlert(error.message);
      });
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
      statusReady = false;
      setLocked(true);
      showAlert(`상태 조회 실패: ${error.message}`);
    }
  }

  async function startFid() {
    showAlert("");
    const itemIds = selectedItemIds();
    if (itemIds.length < MIN_FID_ITEMS || itemIds.length > MAX_FID_ITEMS) {
      showAlert(`${MIN_FID_ITEMS}장 이상 ${MAX_FID_ITEMS}장 이하로 선택해야 한다.`);
      return;
    }
    activeFidResult = null;
    activeFidRunId = null;
    activeFidTotal = itemIds.length;
    showEmptyResult("선택 데이터 측정 중");
    renderFidProgress({state: "running", done: 0, total: activeFidTotal});
    setLocked(true);
    try {
      const response = await fetch(apiUrl("/api/fid/try"), {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({item_ids: itemIds}),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(
          payload.detail ||
          (payload.run_id
            ? `다른 실행이 GPU를 사용 중이다: ${payload.run_id}`
            : `selected FID ${response.status}`),
        );
      }
      activeFidRunId = payload.run_id;
      previousState = "running";
      previousStatusRunId = payload.run_id;
      applyStatus({
        state: "running",
        done: 0,
        total: activeFidTotal,
        log: [],
        run_id: payload.run_id,
      });
    } catch (error) {
      activeFidRunId = null;
      fidProgress.hidden = true;
      showEmptyResult("선택 실행 실패");
      showAlert(error.message);
      await loadStatus();
    }
  }

  if (!Number.isInteger(config.backendPort)) {
    showAlert("BACKEND_PORT 설정을 읽지 못했다.");
    return;
  }

  const apiBase = `${window.location.protocol}//${window.location.hostname}:${config.backendPort}`;
  function apiUrl(path, parameters = {}) {
    const url = new URL(path, apiBase);
    for (const [key, value] of Object.entries(parameters)) url.searchParams.set(key, value);
    return url;
  }

  selectionFillButton.addEventListener("click", () => {
    selectionCheckboxes().slice(0, MAX_FID_ITEMS).forEach((checkbox) => {
      checkbox.checked = true;
    });
    updateSelection();
  });
  selectionResetButton.addEventListener("click", () => {
    selectionCheckboxes().forEach((checkbox) => {
      checkbox.checked = false;
    });
    updateSelection();
  });
  runButton.addEventListener("click", startFid);
  showEmptyResult("선택 실행 대기");
  Promise.all([loadDataset(), loadProtocol(), loadStatus()])
    .catch((error) => showAlert(error.message));
  window.setInterval(loadStatus, 1000);
})();
}
