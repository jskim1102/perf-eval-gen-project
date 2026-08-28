const PERF_RUN_PROGRESS = (() => {
  "use strict";

  const STATE_LABELS = {
    idle: "대기",
    running: "준비 중",
    generating: "이미지 생성 중",
    measuring: "지표 측정 중",
    done: "완료",
    error: "실패",
  };

  function nonNegativeNumber(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0
      ? value : 0;
  }

  function positiveNumber(value) {
    return typeof value === "number" && Number.isFinite(value) && value > 0
      ? value : 0;
  }

  function snapshot(status, expectedTotal = 0) {
    const state = typeof status?.state === "string" ? status.state : "unknown";
    const total = positiveNumber(status?.total) || positiveNumber(expectedTotal);
    const reportedDone = nonNegativeNumber(status?.done);
    const done = total ? Math.min(reportedDone, total) : reportedDone;
    const measuredPercent = total ? Math.round((done / total) * 100) : 0;
    const percent = state === "done" ? 100 : Math.min(99, measuredPercent);
    return {
      state,
      label: STATE_LABELS[state] || state,
      done: state === "done" && total ? total : done,
      total,
      percent,
    };
  }

  return {snapshot};
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = PERF_RUN_PROGRESS;
}
