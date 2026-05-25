const comparisonState = {
  transactionFile: null,
  attendanceFile: null,
  rows: [],
  columns: [],
  currentPage: 1,
  totalPages: 1,
  filteredRowCount: 0,
  pageSize: 25,
  downloadUrl: "",
  debugDownloadUrl: "",
  outputFileName: "",
  timerStartedAt: null,
  timerIntervalId: null,
  searchDebounceId: null,
  pageRequestAbortController: null,
  pageRequestSequence: 0,
};

const comparisonElements = {
  form: document.getElementById("comparison-form"),
  transactionFileInput: document.getElementById("transaction-file-input"),
  attendanceFileInput: document.getElementById("attendance-file-input"),
  transactionFileName: document.getElementById("transaction-file-name"),
  attendanceFileName: document.getElementById("attendance-file-name"),
  compareButton: document.getElementById("compare-button"),
  timer: document.getElementById("comparison-timer"),
  timerValue: document.getElementById("comparison-timer-value"),
  loading: document.getElementById("comparison-loading"),
  error: document.getElementById("comparison-error"),
  resultsSection: document.getElementById("comparison-results-section"),
  summaryGrid: document.getElementById("comparison-summary-grid"),
  searchInput: document.getElementById("comparison-search-input"),
  resultFilter: document.getElementById("comparison-result-filter"),
  resultCount: document.getElementById("comparison-result-count"),
  pageIndicator: document.getElementById("comparison-page-indicator"),
  prevPage: document.getElementById("comparison-prev-page"),
  nextPage: document.getElementById("comparison-next-page"),
  downloadButton: document.getElementById("comparison-download-button"),
  downloadDebugButton: document.getElementById("comparison-download-debug-button"),
  tableHead: document.getElementById("comparison-table-head"),
  tableBody: document.getElementById("comparison-table-body"),
  emptyState: document.getElementById("comparison-empty-state"),
};

const comparisonRequiredElements = [
  "form",
  "transactionFileInput",
  "attendanceFileInput",
  "transactionFileName",
  "attendanceFileName",
  "compareButton",
  "timer",
  "timerValue",
  "loading",
  "error",
  "resultsSection",
  "summaryGrid",
  "searchInput",
  "resultFilter",
  "resultCount",
  "pageIndicator",
  "prevPage",
  "nextPage",
  "downloadButton",
  "downloadDebugButton",
  "tableHead",
  "tableBody",
  "emptyState",
];

const comparisonColumns = [
  "employee_id",
  "date",
  "attendance_punch_in_time",
  "attendance_punch_out_time",
  "transaction_first_in_time",
  "transaction_last_out_time",
  "attendance_truth_shift_family",
  "predicted_final_shift",
  "comparison_result",
  "mismatch_reason",
];

const comparisonLabels = {
  employee_id: "Employee ID",
  date: "Date",
  prediction_row_found: "Prediction Row Found",
  attendance_punch_in_time: "Attendance In",
  attendance_punch_out_time: "Attendance Out",
  transaction_first_in_time: "Transaction First In",
  transaction_last_out_time: "Transaction Last Out",
  attendance_truth_shift_family: "Attendance Shift",
  predicted_final_shift: "Predicted Final Shift",
  predicted_final_status_label: "Predicted Final Status",
  predicted_final_day_status: "Predicted Final Day Status",
  attendance_truth_shift: "Attendance Raw Shift",
  attendance_truth_status: "Attendance Truth Status",
  comparison_layer: "Comparison Layer",
  comparison_result: "Comparison Result",
  mismatch_reason: "Mismatch Reason",
};

const summarySections = [
  {
    title: "Coverage",
    cards: [
      { key: "attendance_total_rows", label: "Attendance Rows", tone: "total", formatter: String },
      { key: "prediction_total_rows", label: "Prediction Rows", tone: "total", formatter: String },
      { key: "matched_rows", label: "Matched Rows", tone: "good", formatter: String },
      { key: "missing_in_prediction", label: "Missing In Prediction", tone: "withheld", formatter: String },
      { key: "missing_in_attendance", label: "Missing In Attendance", tone: "wo", formatter: String },
      { key: "coverage_percent", label: "Coverage", tone: "good", formatter: formatPercent },
    ],
  },
  {
    title: "Shift Answer-Key Accuracy",
    cards: [
      { key: "comparable_shift_rows", label: "Comparable Shift Rows", tone: "total", formatter: String },
      { key: "exact_shift_matches", label: "Exact Matches", tone: "good", formatter: String },
      { key: "exact_shift_mismatches", label: "Exact Mismatches", tone: "withheld", formatter: String },
      { key: "review_shift_matches", label: "Review Matches", tone: "review", formatter: String },
      { key: "review_shift_mismatches", label: "Review Mismatches", tone: "review", formatter: String },
      { key: "predicted_only_shift_accuracy_percent", label: "Shift Accuracy % (Predicted Only)", tone: "good", formatter: formatPercent },
      { key: "assigned_shift_accuracy_percent", label: "Shift Accuracy % (Predicted + Review)", tone: "good", formatter: formatPercent },
      { key: "non_working_excluded", label: "Non-Working Excluded", tone: "wo", formatter: String },
    ],
  },
];

const comparisonReady = validateComparisonElements();
if (comparisonReady) {
  initializeComparisonPage();
}

function validateComparisonElements() {
  const missing = comparisonRequiredElements.filter((key) => !comparisonElements[key]);
  if (missing.length === 0) {
    return true;
  }

  const message = `Comparison UI initialization failed. Missing DOM element(s): ${missing.join(", ")}.`;
  console.error(message);
  if (comparisonElements.error) {
    comparisonElements.error.hidden = false;
    comparisonElements.error.textContent = message;
  }
  return false;
}

function initializeComparisonPage() {
  comparisonElements.transactionFileInput.addEventListener("change", handleTransactionSelection);
  comparisonElements.attendanceFileInput.addEventListener("change", handleAttendanceSelection);
  comparisonElements.form.addEventListener("submit", submitComparison);
  comparisonElements.searchInput.addEventListener("input", scheduleComparisonPageRefresh);
  comparisonElements.resultFilter.addEventListener("change", () => refreshComparisonPage({ page: 1 }));
  comparisonElements.prevPage.addEventListener("click", () => changeComparisonPage(-1));
  comparisonElements.nextPage.addEventListener("click", () => changeComparisonPage(1));
}

function handleTransactionSelection(event) {
  const [file] = event.target.files;
  comparisonState.transactionFile = file || null;
  comparisonElements.transactionFileName.textContent = file ? `Selected: ${file.name}` : "No transaction file selected";
  hideComparisonError();
}

function handleAttendanceSelection(event) {
  const [file] = event.target.files;
  comparisonState.attendanceFile = file || null;
  comparisonElements.attendanceFileName.textContent = file ? `Selected: ${file.name}` : "No attendance file selected";
  hideComparisonError();
}

async function submitComparison(event) {
  event.preventDefault();
  hideComparisonError();

  if (!comparisonState.transactionFile) {
    showComparisonError("Choose a raw transaction file before running comparison.");
    return;
  }

  if (!comparisonState.attendanceFile) {
    showComparisonError("Choose an attendance truth file before running comparison.");
    return;
  }

  cancelPendingComparisonPageRequest();
  setComparisonLoading(true);
  startComparisonTimer();
  comparisonElements.resultsSection.hidden = true;

  try {
    const formData = new FormData();
    formData.append("transaction_file", comparisonState.transactionFile);
    formData.append("attendance_file", comparisonState.attendanceFile);

    const response = await fetch("/api/compare", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      showComparisonError(payload.detail || "Comparison failed.");
      stopComparisonTimer();
      return;
    }

    comparisonState.downloadUrl = payload.download_url || "";
    comparisonState.debugDownloadUrl = payload.debug_download_url || "";
    comparisonState.outputFileName = payload.output_file_name || "comparison.csv";

    renderComparisonSummary(payload.summary || {});
    renderComparisonDownload();
    applyComparisonPayload(payload, { updateFilters: true });
    comparisonElements.resultsSection.hidden = false;
    stopComparisonTimer();
  } catch (error) {
    showComparisonError(error instanceof Error ? error.message : "Comparison failed.");
    stopComparisonTimer();
  } finally {
    setComparisonLoading(false);
  }
}

function applyComparisonPayload(payload, { updateFilters = false } = {}) {
  comparisonState.rows = Array.isArray(payload.data) ? payload.data : [];
  comparisonState.columns = Array.isArray(payload.columns) ? payload.columns : comparisonColumns;
  comparisonState.outputFileName = payload.output_file_name || comparisonState.outputFileName;

  const currentPage = Number(payload.page);
  comparisonState.currentPage = Number.isFinite(currentPage) && currentPage > 0 ? currentPage : 1;

  const pageSize = Number(payload.page_size);
  comparisonState.pageSize = Number.isFinite(pageSize) && pageSize > 0 ? pageSize : comparisonState.pageSize;

  const totalPages = Number(payload.total_pages);
  comparisonState.totalPages = Number.isFinite(totalPages) && totalPages > 0 ? totalPages : 1;

  const filteredRowCount = Number(payload.filtered_row_count);
  comparisonState.filteredRowCount = Number.isFinite(filteredRowCount) ? filteredRowCount : comparisonState.rows.length;

  if (updateFilters) {
    const filterOptions = payload.filters || {};
    populateComparisonResultFilter(Array.isArray(filterOptions.results) ? filterOptions.results : []);
  }

  renderComparisonTable();
}

function renderComparisonSummary(summary) {
  comparisonElements.summaryGrid.innerHTML = summarySections
    .map((section) => `
      <section class="comparison-summary-section">
        <h3>${escapeHtml(section.title)}</h3>
        <div class="comparison-summary-cards">
          ${section.cards.map(({ key, label, tone, formatter }) => `
            <article class="summary-card summary-card-${tone}">
              <span>${escapeHtml(label)}</span>
              <strong>${escapeHtml(formatter(summary[key] ?? 0))}</strong>
            </article>
          `).join("")}
        </div>
      </section>
    `)
    .join("");
}

function populateComparisonResultFilter(results) {
  const currentValue = comparisonElements.resultFilter.value;
  comparisonElements.resultFilter.innerHTML = [
    '<option value="ALL">All results</option>',
    ...results.map((result) => `<option value="${escapeHtml(result)}">${escapeHtml(result)}</option>`),
  ].join("");

  comparisonElements.resultFilter.value = results.includes(currentValue) ? currentValue : "ALL";
}

function renderComparisonDownload() {
  if (!comparisonState.downloadUrl) {
    comparisonElements.downloadButton.hidden = true;
    comparisonElements.downloadButton.removeAttribute("href");
    comparisonElements.downloadButton.removeAttribute("download");
    comparisonElements.downloadDebugButton.hidden = true;
    comparisonElements.downloadDebugButton.removeAttribute("href");
    comparisonElements.downloadDebugButton.removeAttribute("download");
    return;
  }

  comparisonElements.downloadButton.hidden = false;
  comparisonElements.downloadButton.href = comparisonState.downloadUrl;
  comparisonElements.downloadButton.setAttribute(
    "download",
    fileNameFromDownloadUrl(comparisonState.downloadUrl, comparisonState.outputFileName),
  );

  if (!comparisonState.debugDownloadUrl) {
    comparisonElements.downloadDebugButton.hidden = true;
    comparisonElements.downloadDebugButton.removeAttribute("href");
    comparisonElements.downloadDebugButton.removeAttribute("download");
    return;
  }

  comparisonElements.downloadDebugButton.hidden = false;
  comparisonElements.downloadDebugButton.href = comparisonState.debugDownloadUrl;
  comparisonElements.downloadDebugButton.setAttribute(
    "download",
    fileNameFromDownloadUrl(comparisonState.debugDownloadUrl, comparisonState.outputFileName),
  );
}

function fileNameFromDownloadUrl(url, fallbackName) {
  const path = String(url || "").split("?")[0];
  const fileName = decodeURIComponent(path.split("/").filter(Boolean).pop() || "");
  return fileName || fallbackName;
}

function scheduleComparisonPageRefresh() {
  if (comparisonState.searchDebounceId !== null) {
    window.clearTimeout(comparisonState.searchDebounceId);
  }

  comparisonState.searchDebounceId = window.setTimeout(() => {
    comparisonState.searchDebounceId = null;
    refreshComparisonPage({ page: 1 });
  }, 250);
}

function cancelPendingComparisonPageRequest() {
  if (comparisonState.searchDebounceId !== null) {
    window.clearTimeout(comparisonState.searchDebounceId);
    comparisonState.searchDebounceId = null;
  }

  if (comparisonState.pageRequestAbortController) {
    comparisonState.pageRequestAbortController.abort();
    comparisonState.pageRequestAbortController = null;
  }
}

async function refreshComparisonPage({ page = comparisonState.currentPage } = {}) {
  if (!comparisonState.outputFileName) {
    return;
  }

  cancelPendingComparisonPageRequest();

  const requestId = comparisonState.pageRequestSequence + 1;
  comparisonState.pageRequestSequence = requestId;
  const controller = new AbortController();
  comparisonState.pageRequestAbortController = controller;

  const params = new URLSearchParams({
    page: String(page),
    page_size: String(comparisonState.pageSize),
    search: comparisonElements.searchInput.value.trim(),
    result: comparisonElements.resultFilter.value,
  });

  try {
    const response = await fetch(`/api/comparison-results/${encodeURIComponent(comparisonState.outputFileName)}?${params.toString()}`, {
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => ({}));
    if (requestId !== comparisonState.pageRequestSequence) {
      return;
    }
    if (!response.ok) {
      showComparisonError(payload.detail || "Comparison results could not be loaded.");
      return;
    }

    hideComparisonError();
    applyComparisonPayload(payload, { updateFilters: true });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return;
    }
    showComparisonError(error instanceof Error ? error.message : "Comparison results could not be loaded.");
  } finally {
    if (comparisonState.pageRequestAbortController === controller) {
      comparisonState.pageRequestAbortController = null;
    }
  }
}

function renderComparisonTable() {
  const visibleColumns = comparisonColumns.filter((column) => comparisonState.columns.includes(column));

  comparisonElements.tableHead.innerHTML = `
    <tr>
      ${visibleColumns.map((column) => `<th class="${getColumnClass(column)}">${escapeHtml(getComparisonLabel(column))}</th>`).join("")}
    </tr>
  `;

  comparisonElements.tableBody.innerHTML = comparisonState.rows
    .map((row) => `
      <tr class="result-${String(row.comparison_result || "").toLowerCase()}">
        ${visibleColumns.map((column) => `<td class="${getColumnClass(column)}">${formatComparisonCell(row[column], column)}</td>`).join("")}
      </tr>
    `)
    .join("");

  comparisonElements.emptyState.hidden = comparisonState.rows.length > 0;
  comparisonElements.resultCount.textContent = `${comparisonState.filteredRowCount} row${comparisonState.filteredRowCount === 1 ? "" : "s"}`;
  comparisonElements.pageIndicator.textContent = `Page ${comparisonState.currentPage} of ${comparisonState.totalPages}`;
  comparisonElements.prevPage.disabled = comparisonState.currentPage <= 1;
  comparisonElements.nextPage.disabled = comparisonState.currentPage >= comparisonState.totalPages;
}

function changeComparisonPage(direction) {
  const nextPage = comparisonState.currentPage + direction;
  if (nextPage < 1 || nextPage > comparisonState.totalPages) {
    return;
  }

  refreshComparisonPage({ page: nextPage });
}

function formatComparisonCell(value, column) {
  if (value === null || value === undefined || value === "") {
    return '<span class="cell-text-muted">—</span>';
  }

  if (column === "comparison_result") {
    const result = String(value);
    return `<span class="comparison-result-badge comparison-result-${result.toLowerCase()}">${escapeHtml(result)}</span>`;
  }

  if (column === "comparison_layer") {
    return `<span class="comparison-layer-badge">${escapeHtml(String(value))}</span>`;
  }

  if (["predicted_final_shift", "attendance_truth_shift", "attendance_truth_shift_family"].includes(column)) {
    return `<span class="cell-code">${escapeHtml(String(value))}</span>`;
  }

  if ([
    "attendance_punch_in_time",
    "attendance_punch_out_time",
    "transaction_first_in_time",
    "transaction_last_out_time",
  ].includes(column)) {
    return `<span class="cell-code">${escapeHtml(String(value))}</span>`;
  }

  if (column === "mismatch_reason") {
    const text = escapeHtml(String(value));
    return `<span class="cell-truncate" title="${text}">${text}</span>`;
  }

  return escapeHtml(String(value));
}

function getComparisonLabel(column) {
  return comparisonLabels[column] || column.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function getColumnClass(column) {
  return `col-${column.toLowerCase()}`;
}

function setComparisonLoading(isLoading) {
  comparisonElements.compareButton.disabled = isLoading;
  comparisonElements.loading.hidden = !isLoading;
  comparisonElements.compareButton.textContent = isLoading ? "Running..." : "Run Prediction + Comparison";
}

function startComparisonTimer() {
  stopComparisonTimer();
  comparisonState.timerStartedAt = performance.now();
  updateComparisonTimer();
  comparisonElements.timer.classList.add("is-running");
  comparisonState.timerIntervalId = window.setInterval(updateComparisonTimer, 100);
}

function stopComparisonTimer() {
  if (comparisonState.timerIntervalId !== null) {
    window.clearInterval(comparisonState.timerIntervalId);
    comparisonState.timerIntervalId = null;
  }

  updateComparisonTimer();
  comparisonElements.timer.classList.remove("is-running");
}

function updateComparisonTimer() {
  if (comparisonState.timerStartedAt === null) {
    comparisonElements.timerValue.textContent = "00:00.0";
    return;
  }

  comparisonElements.timerValue.textContent = formatElapsedTime(
    performance.now() - comparisonState.timerStartedAt,
  );
}

function formatElapsedTime(elapsedMs) {
  const totalTenths = Math.max(0, Math.floor(elapsedMs / 100));
  const minutes = Math.floor(totalTenths / 600);
  const seconds = Math.floor((totalTenths % 600) / 10);
  const tenths = totalTenths % 10;

  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${tenths}`;
}

function formatPercent(value) {
  return `${Number(value || 0).toFixed(2)}%`;
}

function showComparisonError(message) {
  comparisonElements.error.hidden = false;
  comparisonElements.error.textContent = String(message || "Comparison failed.");
}

function hideComparisonError() {
  comparisonElements.error.hidden = true;
  comparisonElements.error.textContent = "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
