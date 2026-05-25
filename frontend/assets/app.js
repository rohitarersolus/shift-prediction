const state = {
  file: null,
  rows: [],
  columns: [],
  currentPage: 1,
  pageSize: 20,
  totalPages: 1,
  totalRows: 0,
  filteredRowCount: 0,
  downloadUrl: "",
  debugDownloadUrl: "",
  outputFileName: "",
  resultFileName: "",
  selectedRow: null,
  predictionTimerStartedAt: null,
  predictionTimerIntervalId: null,
  searchDebounceId: null,
  pageRequestAbortController: null,
  pageRequestSequence: 0,
  manualPredictionLoading: false,
};

const candidateSliceLabel = "Candidate Punch Slice";

const elements = {
  form: document.getElementById("prediction-form"),
  fileInput: document.getElementById("file-input"),
  dropzone: document.getElementById("dropzone"),
  selectedFileName: document.getElementById("selected-file-name"),
  predictButton: document.getElementById("predict-button"),
  loadingIndicator: document.getElementById("loading-indicator"),
  predictionTimer: document.getElementById("prediction-timer"),
  predictionTimerValue: document.getElementById("prediction-timer-value"),
  errorBanner: document.getElementById("error-banner"),
  resultsSection: document.getElementById("results-section"),
  summaryGrid: document.getElementById("summary-grid"),
  searchInput: document.getElementById("search-input"),
  statusFilter: document.getElementById("status-filter"),
  shiftFilter: document.getElementById("shift-filter"),
  mismatchFilter: document.getElementById("mismatch-filter"),
  resultCount: document.getElementById("result-count"),
  pageIndicator: document.getElementById("page-indicator"),
  tableHead: document.getElementById("table-head"),
  tableBody: document.getElementById("table-body"),
  emptyState: document.getElementById("empty-state"),
  prevPage: document.getElementById("prev-page"),
  nextPage: document.getElementById("next-page"),
  downloadButton: document.getElementById("download-button"),
  downloadDebugButton: document.getElementById("download-debug-button"),
  detailPanel: document.getElementById("detail-panel"),
  manualPredictOpen: document.getElementById("manual-predict-open"),
  manualPredictDialog: document.getElementById("manual-predict-dialog"),
  manualPredictForm: document.getElementById("manual-predict-form"),
  manualPredictClose: document.getElementById("manual-predict-close"),
  manualEmployeeId: document.getElementById("manual-employee-id"),
  manualDate: document.getElementById("manual-date"),
  manualPunchIn: document.getElementById("manual-punch-in"),
  manualPunchOut: document.getElementById("manual-punch-out"),
  manualExtraPunches: document.getElementById("manual-extra-punches"),
  manualNote: document.getElementById("manual-note"),
  manualError: document.getElementById("manual-error"),
  manualPredictSubmit: document.getElementById("manual-predict-submit"),
  manualPredictClear: document.getElementById("manual-predict-clear"),
  manualResult: document.getElementById("manual-result"),
};

const requiredElementKeys = [
  "form",
  "fileInput",
  "dropzone",
  "selectedFileName",
  "predictButton",
  "loadingIndicator",
  "predictionTimer",
  "predictionTimerValue",
  "errorBanner",
  "resultsSection",
  "summaryGrid",
  "searchInput",
  "statusFilter",
  "shiftFilter",
  "mismatchFilter",
  "resultCount",
  "pageIndicator",
  "tableHead",
  "tableBody",
  "emptyState",
  "prevPage",
  "nextPage",
  "downloadButton",
  "downloadDebugButton",
];

const missingElementWarnings = new Set();

const summaryKeys = [
  { key: "total_rows", label: "Total Rows", tone: "total" },
  { key: "SHIFT_PREDICTED", label: "Shift Predicted", tone: "good" },
  { key: "SHIFT_REVIEW", label: "Shift Review", tone: "review" },
  { key: "ABSENT", label: "Absent", tone: "absent" },
  { key: "WO_REVIEW", label: "Worked on WO", tone: "wo" },
  { key: "WITHHELD", label: "Withheld", tone: "withheld" },
];

const preferredVisibleColumns = [
  "EmpCode_norm",
  "attendance_day",
  "punch_in_time",
  "punch_out_time",
  "txn_punch_count",
  "final_shift",
  "final_status_label",
  "shift_status",
  "final_message",
  "prediction_mismatch_status",
  "prediction_mismatch_message",
  "raw_transactions",
];

const primaryDetailFields = [
  "prod_model_pred_shift",
  "prod_model_confidence",
  "decision_reason",
];

const pairDetailFields = [
  "valid_reader_pair_found",
  "window_punch_count",
  "pair_span_min",
  "reader_pair_method",
  "reader_pair_out_scope",
  "pair_ignored_after_out_count",
];

const detailFieldLabels = {
  EmpCode_norm: "EmpCode Norm",
  attendance_day: "Attendance Day",
  weekday_num: "Weekday Num",
  is_sunday: "Is Sunday",
  best_shift_candidate: "Best Shift Candidate",
  prod_model_pred_shift: "Model Predicted Shift",
  prod_model_confidence: "Model Confidence",
  working_shift_hint: "Working Shift Hint",
  final_day_status: "Final Day Status",
  final_shift: "Final Shift",
  shift_status: "Shift Status",
  final_status_label: "Final Status Label",
  final_message: "Final Message",
  prediction_mismatch_status: "Mismatch Status",
  prediction_mismatch_message: "Mismatch Message",
  window_punch_count: "Window Punch Count",
  pair_punch_count: "Candidate Punch Count",
  valid_pair_found: "Valid Reader Pair Found",
  pair_confidence_bucket: "Candidate Confidence Bucket",
  pair_start_min: "Candidate Start Min",
  pair_end_min: "Candidate End Min",
  pair_span_min: "Pair Span",
  reader_pair_method: "Reader Pair Method",
  reader_pair_out_scope: "Reader Pair Out Scope",
  pair_ignored_after_out_count: "Ignored After Out Count",
  pair_next_day_flag: "Candidate Next Day Flag",
  pair_preview: candidateSliceLabel,
  raw_transactions: "Raw Transactions",
  punch_in_time: "Punch In",
  punch_out_time: "Punch Out",
  txn_punch_count: "Same-Day Punch Count",
  valid_reader_pair_found: "Valid Reader Pair Found",
  reader_pair_found_flag: "Valid Reader Pair Found",
  decision_reason: "Decision Reason",
};

const appReady = validateRequiredElements();
if (appReady) {
  bindEvents();
}

function validateRequiredElements() {
  const missingKeys = requiredElementKeys.filter((key) => !elements[key]);
  if (missingKeys.length === 0) {
    return true;
  }

  const message = `Shift Prediction UI initialization failed. Missing DOM element(s): ${missingKeys.join(", ")}.`;
  console.error(message);

  const errorBanner = elements.errorBanner;
  if (errorBanner) {
    errorBanner.hidden = false;
    errorBanner.textContent = message;
  }

  return false;
}

function getElement(key) {
  const element = elements[key];
  if (element) {
    return element;
  }

  if (!missingElementWarnings.has(key)) {
    missingElementWarnings.add(key);
    console.error(`Shift Prediction UI missing expected element: ${key}`);
  }

  return null;
}

function setElementHtml(key, html) {
  const element = getElement(key);
  if (!element) {
    return false;
  }

  element.innerHTML = html;
  return true;
}

function setElementText(key, text) {
  const element = getElement(key);
  if (!element) {
    return false;
  }

  element.textContent = text;
  return true;
}

function bindEvents() {
  const form = getElement("form");
  const fileInput = getElement("fileInput");
  const dropzone = getElement("dropzone");
  const searchInput = getElement("searchInput");
  const statusFilter = getElement("statusFilter");
  const shiftFilter = getElement("shiftFilter");
  const mismatchFilter = getElement("mismatchFilter");
  const prevPage = getElement("prevPage");
  const nextPage = getElement("nextPage");
  const tableBody = getElement("tableBody");

  if (!form || !fileInput || !dropzone || !searchInput || !statusFilter || !shiftFilter || !mismatchFilter || !prevPage || !nextPage || !tableBody) {
    return;
  }

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileSelection);
  form.addEventListener("submit", submitPrediction);
  searchInput.addEventListener("input", schedulePredictionPageRefresh);
  statusFilter.addEventListener("change", () => refreshPredictionPage({ page: 1 }));
  shiftFilter.addEventListener("change", () => refreshPredictionPage({ page: 1 }));
  mismatchFilter.addEventListener("change", () => refreshPredictionPage({ page: 1 }));
  prevPage.addEventListener("click", () => changePage(-1));
  nextPage.addEventListener("click", () => changePage(1));
  tableBody.addEventListener("click", handleRowSelection);
  bindManualPredictionEvents();

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (!file) {
      return;
    }

    fileInput.files = event.dataTransfer.files;
    setSelectedFile(file);
  });
}

function bindManualPredictionEvents() {
  const openButton = elements.manualPredictOpen;
  const dialog = elements.manualPredictDialog;
  const form = elements.manualPredictForm;
  const closeButton = elements.manualPredictClose;
  const clearButton = elements.manualPredictClear;

  if (!openButton || !dialog || !form || !closeButton || !clearButton) {
    return;
  }

  openButton.addEventListener("click", openManualPredictionDialog);
  closeButton.addEventListener("click", closeManualPredictionDialog);
  clearButton.addEventListener("click", clearManualPredictionForm);
  form.addEventListener("submit", submitManualPrediction);

  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      closeManualPredictionDialog();
    }
  });
}

function openManualPredictionDialog() {
  hideManualError();
  if (elements.manualPredictDialog.showModal) {
    elements.manualPredictDialog.showModal();
  } else {
    elements.manualPredictDialog.setAttribute("open", "");
  }

  if (!elements.manualDate.value) {
    elements.manualDate.value = formatLocalInputDate(new Date());
  }
  elements.manualEmployeeId.focus();
}

function closeManualPredictionDialog() {
  elements.manualPredictDialog.close();
}

function clearManualPredictionForm() {
  elements.manualPredictForm.reset();
  elements.manualResult.hidden = true;
  elements.manualResult.innerHTML = "";
  hideManualError();
  setManualLoading(false);
}

async function submitManualPrediction(event) {
  event.preventDefault();
  hideManualError();
  elements.manualResult.hidden = true;

  const payload = {
    employee_id: elements.manualEmployeeId.value.trim(),
    date: elements.manualDate.value,
    punch_in: normalizeManualDateTimeInput(elements.manualPunchIn.value),
    punch_out: normalizeManualDateTimeInput(elements.manualPunchOut.value),
    extra_punches: elements.manualExtraPunches.value.trim(),
    note: elements.manualNote.value.trim(),
  };

  if (!payload.employee_id || !payload.date || !payload.punch_in || !payload.punch_out) {
    showManualError("Employee ID, Date, Punch In, and Punch Out are required.");
    return;
  }

  setManualLoading(true);

  try {
    const response = await fetch("/api/manual-predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      showManualError(result.detail || "Manual prediction failed.");
      return;
    }

    renderManualPredictionResult(result);
  } catch (error) {
    showManualError(error instanceof Error ? error.message : "Manual prediction failed.");
  } finally {
    setManualLoading(false);
  }
}

function renderManualPredictionResult(result) {
  const historyText = result.history_confidence === null || result.history_confidence === undefined
    ? "—"
    : `${formatPercentage(result.history_confidence)}${result.history_consistency ? ` (${result.history_consistency})` : ""}`;

  elements.manualResult.innerHTML = `
    <div class="manual-result-header">
      <div>
        <p class="detail-kicker">Manual Test Result</p>
        <h3>${escapeHtml(result.employee_id || "Employee")}</h3>
        <p>${escapeHtml(formatDateValue(result.date || ""))}</p>
      </div>
      ${statusChip(result.status || "review")}
    </div>

    <div class="manual-result-grid">
      ${manualResultItem("Model Predicted Shift", result.model_predicted_shift || "unknown", true)}
      ${manualResultItem("Model Confidence", result.model_confidence === null || result.model_confidence === undefined ? "—" : formatPercentage(result.model_confidence))}
      ${manualResultItem("Historical Regular Shift", result.historical_regular_shift || "unknown", true)}
      ${manualResultItem("History Confidence", historyText)}
      ${manualResultItem("Final Recommended Shift", result.final_recommended_shift || "unknown", true)}
      ${manualResultItem("Status", result.status || "review")}
      ${manualResultItem("Message", result.message || "—", false, "manual-result-wide")}
    </div>

    <article class="manual-raw">
      <span>Raw transactions used</span>
      <div>${formatManualTransactions(result.raw_transactions_used)}</div>
    </article>
  `;
  elements.manualResult.hidden = false;
}

function manualResultItem(label, value, code = false, extraClass = "") {
  return `
    <article class="manual-result-item ${extraClass}">
      <span>${escapeHtml(label)}</span>
      <strong>${code ? `<span class="cell-code">${escapeHtml(String(value))}</span>` : escapeHtml(String(value))}</strong>
    </article>
  `;
}

function formatManualTransactions(transactions) {
  if (!Array.isArray(transactions) || transactions.length === 0) {
    return '<span class="cell-text-muted">—</span>';
  }

  return transactions.map((transaction) => escapeHtml(String(transaction))).join("<br>");
}

function setManualLoading(isLoading) {
  state.manualPredictionLoading = isLoading;
  if (!elements.manualPredictSubmit) {
    return;
  }

  elements.manualPredictSubmit.disabled = isLoading;
  elements.manualPredictSubmit.textContent = isLoading ? "Predicting..." : "Predict";
}

function showManualError(message) {
  if (!elements.manualError) {
    return;
  }

  elements.manualError.hidden = false;
  elements.manualError.textContent = message;
}

function hideManualError() {
  if (!elements.manualError) {
    return;
  }

  elements.manualError.hidden = true;
  elements.manualError.textContent = "";
}

function handleFileSelection(event) {
  const [file] = event.target.files;
  setSelectedFile(file || null);
}

function setSelectedFile(file) {
  if (!appReady) {
    return;
  }

  state.file = file;
  setElementText("selectedFileName", file ? file.name : "No file selected");
  hideError();
}

function resetPredictionFilters() {
  const searchInput = getElement("searchInput");
  const statusFilter = getElement("statusFilter");
  const shiftFilter = getElement("shiftFilter");
  const mismatchFilter = getElement("mismatchFilter");

  if (searchInput) {
    searchInput.value = "";
  }
  if (statusFilter) {
    statusFilter.value = "ALL";
  }
  if (shiftFilter) {
    shiftFilter.value = "ALL";
  }
  if (mismatchFilter) {
    mismatchFilter.value = "NONE";
  }
}

async function submitPrediction(event) {
  event.preventDefault();
  if (!appReady) {
    return;
  }

  hideError();

  if (!state.file) {
    showError("Choose a .xls, .xlsx, or .csv file before running prediction.");
    return;
  }

  const extension = `.${state.file.name.split(".").pop().toLowerCase()}`;
  if (![".xls", ".xlsx", ".csv"].includes(extension)) {
    showError("Unsupported file type. Please upload a .xls, .xlsx, or .csv file.");
    return;
  }

  setLoading(true);
  startPredictionTimer();
  hideResults();
  cancelPendingPageRequest();

  try {
    const formData = new FormData();
    formData.append("file", state.file);

    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      showError(payload);
      stopPredictionTimer();
      return;
    }

    state.downloadUrl = payload.download_url || "";
    state.debugDownloadUrl = payload.debug_download_url || "";
    state.outputFileName = payload.output_file_name || "shift-predictions.csv";
    state.resultFileName = payload.output_file_name || "";
    resetPredictionFilters();

    renderSummary(payload.summary || {});
    renderDownloadButton();
    applyPredictionPagePayload(payload, { updateFilters: true });
    showResults();
    stopPredictionTimer();
  } catch (error) {
    showError(error);
    stopPredictionTimer();
  } finally {
    setLoading(false);
  }
}

function applyPredictionPagePayload(payload, { updateFilters = false } = {}) {
  const previousSelectionKey = state.selectedRow ? buildRowSelectionKey(state.selectedRow) : "";

  state.rows = normalizePredictionRows(Array.isArray(payload.data) ? payload.data : []);
  state.columns = buildAvailableColumns(Array.isArray(payload.columns) ? payload.columns : [], state.rows);
  state.currentPage = Number(payload.page) || 1;
  state.pageSize = Number(payload.page_size) || state.pageSize;
  state.totalPages = Math.max(1, Number(payload.total_pages) || 1);
  state.totalRows = Number(payload.total_rows ?? payload.row_count ?? state.rows.length) || 0;
  state.filteredRowCount = Number(payload.filtered_row_count ?? payload.row_count ?? state.rows.length) || 0;
  state.resultFileName = payload.output_file_name || state.resultFileName;

  if (updateFilters) {
    const filterOptions = payload.filters || {};
    populateStatusFilter(Array.isArray(filterOptions.statuses) ? filterOptions.statuses : []);
    populateShiftFilter(Array.isArray(filterOptions.shifts) ? filterOptions.shifts : []);
    populateMismatchFilter(Array.isArray(filterOptions.mismatch_states) ? filterOptions.mismatch_states : []);
  }

  state.selectedRow = state.rows.find((row) => buildRowSelectionKey(row) === previousSelectionKey) || state.rows[0] || null;

  renderTable();
  renderDetailPanel();
}

function buildRowSelectionKey(row) {
  if (!row || typeof row !== "object") {
    return "";
  }

  return [row.EmpCode_norm || row.EmpCode || "", row.attendance_day || ""].join("::");
}

function schedulePredictionPageRefresh() {
  if (state.searchDebounceId !== null) {
    window.clearTimeout(state.searchDebounceId);
  }

  state.searchDebounceId = window.setTimeout(() => {
    state.searchDebounceId = null;
    refreshPredictionPage({ page: 1 });
  }, 250);
}

function cancelPendingPageRequest() {
  if (state.searchDebounceId !== null) {
    window.clearTimeout(state.searchDebounceId);
    state.searchDebounceId = null;
  }

  if (state.pageRequestAbortController) {
    state.pageRequestAbortController.abort();
    state.pageRequestAbortController = null;
  }
}

async function refreshPredictionPage({ page = state.currentPage } = {}) {
  if (!appReady || !state.resultFileName) {
    return;
  }

  const searchInput = getElement("searchInput");
  const statusFilter = getElement("statusFilter");
  const shiftFilter = getElement("shiftFilter");
  const mismatchFilter = getElement("mismatchFilter");
  if (!searchInput || !statusFilter || !shiftFilter || !mismatchFilter) {
    return;
  }

  cancelPendingPageRequest();

  const requestId = state.pageRequestSequence + 1;
  state.pageRequestSequence = requestId;
  const controller = new AbortController();
  state.pageRequestAbortController = controller;

  const params = new URLSearchParams({
    page: String(page),
    page_size: String(state.pageSize),
    search: searchInput.value.trim(),
    status: statusFilter.value,
    shift: shiftFilter.value,
    mismatch: mismatchFilter.value,
  });

  try {
    const response = await fetch(`/api/prediction-results/${encodeURIComponent(state.resultFileName)}?${params.toString()}`, {
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => ({}));
    if (requestId !== state.pageRequestSequence) {
      return;
    }
    if (!response.ok) {
      showError(payload);
      return;
    }

    hideError();
    applyPredictionPagePayload(payload, { updateFilters: true });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return;
    }
    showError(error);
  } finally {
    if (state.pageRequestAbortController === controller) {
      state.pageRequestAbortController = null;
    }
  }
}

function renderSummary(summary) {
  const normalizedSummary = normalizeTransactionSummary(summary);
  const html = summaryKeys
    .map(
      ({ key, label, tone }) => `
        <article class="summary-card summary-card-${tone}">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(String(normalizedSummary[key] ?? 0))}</strong>
        </article>
      `,
    )
    .join("");

  setElementHtml("summaryGrid", html);
}

function normalizeTransactionSummary(summary) {
  const safeSummary = summary || {};

  return {
    total_rows: safeSummary.total_rows ?? 0,
    SHIFT_PREDICTED: safeSummary.SHIFT_PREDICTED ?? 0,
    SHIFT_REVIEW: safeSummary.SHIFT_REVIEW ?? 0,
    ABSENT: safeSummary.ABSENT ?? 0,
    WITHHELD: safeSummary.WITHHELD ?? 0,
    WO_REVIEW: safeSummary.WO_REVIEW ?? 0,
  };
}

function renderDownloadButton() {
  const downloadButton = getElement("downloadButton");
  const downloadDebugButton = getElement("downloadDebugButton");
  if (!downloadButton || !downloadDebugButton) {
    return;
  }

  if (!state.downloadUrl) {
    downloadButton.hidden = true;
    downloadButton.removeAttribute("href");
    downloadButton.removeAttribute("download");
    downloadDebugButton.hidden = true;
    downloadDebugButton.removeAttribute("href");
    downloadDebugButton.removeAttribute("download");
    return;
  }

  downloadButton.hidden = false;
  downloadButton.href = state.downloadUrl;
  downloadButton.setAttribute("download", fileNameFromDownloadUrl(state.downloadUrl, state.outputFileName));

  if (!state.debugDownloadUrl) {
    downloadDebugButton.hidden = true;
    downloadDebugButton.removeAttribute("href");
    downloadDebugButton.removeAttribute("download");
    return;
  }

  downloadDebugButton.hidden = false;
  downloadDebugButton.href = state.debugDownloadUrl;
  downloadDebugButton.setAttribute("download", fileNameFromDownloadUrl(state.debugDownloadUrl, state.outputFileName));
}

function fileNameFromDownloadUrl(url, fallbackName) {
  const path = String(url || "").split("?")[0];
  const fileName = decodeURIComponent(path.split("/").filter(Boolean).pop() || "");
  return fileName || fallbackName;
}

function populateStatusFilter(statuses) {
  const statusFilter = getElement("statusFilter");
  if (!statusFilter) {
    return;
  }

  const currentValue = statusFilter.value;

  statusFilter.innerHTML = [
    '<option value="ALL">All statuses</option>',
    ...statuses.map((status) => `<option value="${escapeHtml(String(status))}">${escapeHtml(String(status))}</option>`),
  ].join("");

  if (statuses.includes(currentValue)) {
    statusFilter.value = currentValue;
  }
}

function populateShiftFilter(shifts) {
  const shiftFilter = getElement("shiftFilter");
  if (!shiftFilter) {
    return;
  }

  const currentValue = shiftFilter.value;

  shiftFilter.innerHTML = [
    '<option value="ALL">All shifts</option>',
    ...shifts.map((shift) => `<option value="${escapeHtml(String(shift))}">${escapeHtml(String(shift))}</option>`),
  ].join("");

  if (shifts.includes(currentValue)) {
    shiftFilter.value = currentValue;
  }
}

function populateMismatchFilter(mismatchStates) {
  const mismatchFilter = getElement("mismatchFilter");
  if (!mismatchFilter) {
    return;
  }

  const currentValue = mismatchFilter.value;

  mismatchFilter.innerHTML = [
    '<option value="NONE">None</option>',
    '<option value="ALL">All mismatch states</option>',
    ...mismatchStates.map((stateValue) => `<option value="${escapeHtml(String(stateValue))}">${escapeHtml(formatStatusLabel(String(stateValue)))}</option>`),
  ].join("");

  if (currentValue === "NONE" || currentValue === "ALL" || mismatchStates.includes(currentValue)) {
    mismatchFilter.value = currentValue;
  }
}

function getVisibleColumns() {
  const visibleColumns = preferredVisibleColumns.filter((column) => (
    state.columns.includes(column) || isDerivedColumn(column)
  ));

  if (visibleColumns.length > 0) {
    return visibleColumns;
  }

  return state.columns.slice(0, 8);
}

function renderTable() {
  const tableHead = getElement("tableHead");
  const tableBody = getElement("tableBody");
  const emptyState = getElement("emptyState");
  const prevPage = getElement("prevPage");
  const nextPage = getElement("nextPage");

  if (!tableHead || !tableBody || !emptyState || !prevPage || !nextPage) {
    return;
  }

  const visibleColumns = getVisibleColumns();
  const totalPages = Math.max(1, state.totalPages);
  state.currentPage = Math.min(state.currentPage, totalPages);
  const pageRows = state.rows;

  tableHead.innerHTML = `
    <tr>
      ${visibleColumns.map((column) => `<th class="${getColumnClass(column)}">${escapeHtml(getFieldLabel(column))}</th>`).join("")}
    </tr>
  `;

  tableBody.innerHTML = pageRows
    .map((row, index) => {
      const selectedClass = row === state.selectedRow ? "is-selected" : "";
      return `
        <tr class="${selectedClass}" data-row-index="${index}" tabindex="0" aria-selected="${row === state.selectedRow}">
          ${visibleColumns.map((column) => `<td class="${getColumnClass(column)}">${formatCell(row[column], column, row)}</td>`).join("")}
        </tr>
      `;
    })
    .join("");

  emptyState.hidden = pageRows.length > 0;
  setElementText("resultCount", `${state.filteredRowCount} row${state.filteredRowCount === 1 ? "" : "s"}`);
  setElementText("pageIndicator", `Page ${state.currentPage} of ${totalPages}`);
  prevPage.disabled = state.currentPage <= 1;
  nextPage.disabled = state.currentPage >= totalPages;
}

function handleRowSelection(event) {
  const rowElement = event.target.closest("tr[data-row-index]");
  if (!rowElement) {
    return;
  }

  const rowIndex = Number(rowElement.dataset.rowIndex);
  const row = state.rows[rowIndex];
  if (!row) {
    return;
  }

  state.selectedRow = row;
  renderTable();
  renderDetailPanel();
}

function renderDetailPanel() {
  const detailPanel = getElement("detailPanel");
  if (!detailPanel) {
    return;
  }

  if (!state.selectedRow) {
    detailPanel.innerHTML = '<div class="detail-empty">Select a row to inspect model signals and advanced fields.</div>';
    return;
  }

  const row = state.selectedRow;
  const probabilityFields = Object.keys(row)
    .filter((key) => key.startsWith("prod_prob_"))
    .sort();

  const additionalFields = Object.keys(row)
    .filter((key) => !preferredVisibleColumns.includes(key))
    .filter((key) => !primaryDetailFields.includes(key))
    .filter((key) => !pairDetailFields.includes(key))
    .filter((key) => key !== "pair_preview")
    .filter((key) => !probabilityFields.includes(key))
    .filter((key) => row[key] !== null && row[key] !== undefined && row[key] !== "")
    .sort();

  detailPanel.innerHTML = `
    <div class="detail-header">
      <div>
        <p class="detail-kicker">Row Details</p>
        <h3 class="detail-title">${escapeHtml(String(row.EmpCode_norm || row["Employee ID"] || "Selected Row"))}</h3>
        <p class="detail-subtitle">${escapeHtml(formatDateValue(row.attendance_day || ""))}</p>
      </div>
      ${statusPill(row.final_status_label || "Unknown")}
    </div>

    <section class="detail-section">
      <h4>Decision</h4>
      <div class="detail-grid">
        ${primaryDetailFields
          .filter((field) => field in row)
          .map((field) => detailItem(getFieldLabel(field), row[field], field))
          .join("")}
      </div>
    </section>

    <section class="detail-section">
      <h4>Transaction Window</h4>
      <p class="detail-note">Raw transactions are same-day/reader-window timestamps. V3 day-first logic uses earliest same-day ReaderNumber 1 IN and latest same-day ReaderNumber 2 OUT first; next-day OUT is fallback only.</p>
      <div class="detail-grid">
        ${pairDetailFields
          .filter((field) => field in row)
          .map((field) => detailItem(getFieldLabel(field), row[field], field))
          .join("")}
      </div>
      <div class="detail-block-stack">
        <article class="detail-block-card">
          <span class="detail-block-label">${escapeHtml(getFieldLabel("raw_transactions"))}</span>
          <div class="detail-block">${formatPreviewBlock(row.raw_transactions)}</div>
        </article>
        <article class="detail-block-card ${isValidPairFound(row) ? "" : "detail-block-card-muted"}">
          <span class="detail-block-label">${escapeHtml(getFieldLabel("pair_preview"))}</span>
          ${isValidPairFound(row) ? "" : '<p class="detail-block-note">No valid reader pair was formed.</p>'}
          <div class="detail-block">${formatCandidateDetailBlock(row)}</div>
        </article>
      </div>
    </section>

    ${probabilityFields.length > 0 ? `
      <section class="detail-section">
        <h4>Model Probabilities</h4>
        <div class="probability-list">
          ${probabilityFields.map((field) => renderProbabilityRow(field, row[field])).join("")}
        </div>
      </section>
    ` : ""}

    ${additionalFields.length > 0 ? `
      <section class="detail-section">
        <h4>Additional Fields</h4>
        <div class="detail-grid">
          ${additionalFields.map((field) => detailItem(getFieldLabel(field), row[field], field)).join("")}
        </div>
      </section>
    ` : ""}
  `;
}

function detailItem(label, value, fieldName) {
  return `
    <article class="detail-item">
      <span class="detail-label">${escapeHtml(label)}</span>
      <span class="detail-value">${formatDetailValue(value, fieldName)}</span>
    </article>
  `;
}

function renderProbabilityRow(field, value) {
  const numericValue = Number(value);
  const safeValue = Number.isFinite(numericValue) ? numericValue : 0;
  const percentage = `${(safeValue * 100).toFixed(2)}%`;

  return `
    <div class="probability-row">
      <div class="probability-meta">
        <span class="probability-label">${escapeHtml(field.replace("prod_prob_", ""))}</span>
        <span class="probability-value">${percentage}</span>
      </div>
      <div class="probability-track">
        <div class="probability-fill" style="width: ${Math.max(0, Math.min(100, safeValue * 100))}%"></div>
      </div>
    </div>
  `;
}

function changePage(direction) {
  if (!appReady) {
    return;
  }

  const totalPages = Math.max(1, state.totalPages);
  const nextPage = state.currentPage + direction;
  if (nextPage < 1 || nextPage > totalPages) {
    return;
  }

  refreshPredictionPage({ page: nextPage });
}

function setLoading(isLoading) {
  const predictButton = getElement("predictButton");
  const loadingIndicator = getElement("loadingIndicator");
  if (!predictButton || !loadingIndicator) {
    return;
  }

  predictButton.disabled = isLoading;
  loadingIndicator.hidden = !isLoading;
  predictButton.textContent = isLoading ? "Predicting..." : "Run Prediction";
}

function startPredictionTimer() {
  stopPredictionTimer();
  state.predictionTimerStartedAt = performance.now();
  updatePredictionTimer();

  const predictionTimer = getElement("predictionTimer");
  if (predictionTimer) {
    predictionTimer.classList.add("is-running");
  }

  state.predictionTimerIntervalId = window.setInterval(updatePredictionTimer, 100);
}

function stopPredictionTimer() {
  if (state.predictionTimerIntervalId !== null) {
    window.clearInterval(state.predictionTimerIntervalId);
    state.predictionTimerIntervalId = null;
  }

  updatePredictionTimer();

  const predictionTimer = getElement("predictionTimer");
  if (predictionTimer) {
    predictionTimer.classList.remove("is-running");
  }
}

function updatePredictionTimer() {
  const predictionTimerValue = getElement("predictionTimerValue");
  if (!predictionTimerValue) {
    return;
  }

  if (state.predictionTimerStartedAt === null) {
    predictionTimerValue.textContent = "00:00.0";
    return;
  }

  predictionTimerValue.textContent = formatElapsedTime(performance.now() - state.predictionTimerStartedAt);
}

function formatElapsedTime(elapsedMs) {
  const totalTenths = Math.max(0, Math.floor(elapsedMs / 100));
  const minutes = Math.floor(totalTenths / 600);
  const seconds = Math.floor((totalTenths % 600) / 10);
  const tenths = totalTenths % 10;

  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${tenths}`;
}

function showResults() {
  const resultsSection = getElement("resultsSection");
  if (!resultsSection) {
    return;
  }

  resultsSection.hidden = false;
}

function hideResults() {
  const resultsSection = getElement("resultsSection");
  if (!resultsSection) {
    return;
  }

  resultsSection.hidden = true;
}

function showError(error) {
  const errorBanner = getElement("errorBanner");
  if (!errorBanner) {
    return;
  }

  const payload =
    typeof error === "string"
      ? { detail: error }
      : error instanceof Error
        ? { detail: error.message }
        : (error || {});

  const parts = [`<div>${escapeHtml(payload.detail || "Prediction failed.")}</div>`];

  if (Array.isArray(payload.detected_columns) && payload.detected_columns.length > 0) {
    parts.push(
      `<div><strong>Detected columns:</strong> ${escapeHtml(payload.detected_columns.join(", "))}</div>`,
    );
  }

  if (payload.supported_aliases && typeof payload.supported_aliases === "object") {
    const aliasLines = Object.entries(payload.supported_aliases)
      .map(([target, aliases]) => `${escapeHtml(target)}: ${escapeHtml((aliases || []).join(", "))}`)
      .join("<br>");

    parts.push(`<div><strong>Supported aliases:</strong><br>${aliasLines}</div>`);
  }

  errorBanner.hidden = false;
  errorBanner.innerHTML = parts.join("");
}

function hideError() {
  const errorBanner = getElement("errorBanner");
  if (!errorBanner) {
    return;
  }

  errorBanner.hidden = true;
  errorBanner.innerHTML = "";
}

function formatHeader(column) {
  return column
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function getFieldLabel(field) {
  return detailFieldLabels[field] || formatHeader(field);
}

function getColumnClass(column) {
  return `col-${column.toLowerCase()}`;
}

function formatCell(value, column, row) {
  if (value === null || value === undefined || value === "") {
    if (column === "pair_preview" && row) {
      return renderCandidatePreview(row);
    }
    return '<span class="cell-text-muted">—</span>';
  }

  if (column === "valid_pair_found" || column === "valid_reader_pair_found" || column === "reader_pair_found_flag") {
    return renderValidPairBadge(row);
  }

  if (column === "final_status_label" || column === "final_day_status" || column === "prediction_mismatch_status") {
    return statusPill(value);
  }

  if (column === "shift_status") {
    return statusChip(value);
  }

  if (["best_shift_candidate", "prod_model_pred_shift", "final_shift"].includes(column)) {
    return `<span class="cell-code">${escapeHtml(String(value))}</span>`;
  }

  if (column === "is_sunday" || column === "pair_next_day_flag") {
    return escapeHtml(Number(value) === 1 ? "Yes" : "No");
  }

  if (column.endsWith("_min")) {
    return escapeHtml(formatMinuteValue(value));
  }

  if (column === "prod_model_confidence") {
    return escapeHtml(formatPercentage(value));
  }

  if (column.startsWith("prod_prob_")) {
    return escapeHtml(formatPercentage(value));
  }

  if (column === "attendance_day" || isIsoDateLike(value)) {
    return escapeHtml(formatDateValue(value));
  }

  if (column === "final_message" || column === "prediction_mismatch_message") {
    const text = escapeHtml(String(value));
    return `<span class="cell-truncate" title="${text}">${text}</span>`;
  }

  if (column === "pair_preview") {
    return renderCandidatePreview(row);
  }

  if (column === "raw_transactions") {
    const text = escapeHtml(String(value));
    return `<span class="cell-preview" title="${text}">${text}</span>`;
  }

  return escapeHtml(String(value));
}

function formatDetailValue(value, fieldName) {
  if (value === null || value === undefined || value === "") {
    return '<span class="cell-text-muted">—</span>';
  }

  if (fieldName === "valid_pair_found" || fieldName === "valid_reader_pair_found" || fieldName === "reader_pair_found_flag") {
    return renderValidPairBadge({ valid_reader_pair_found: value });
  }

  if (fieldName === "final_status_label" || fieldName === "final_day_status" || fieldName === "prediction_mismatch_status") {
    return statusPill(value);
  }

  if (fieldName === "shift_status") {
    return statusChip(value);
  }

  if (fieldName === "prod_model_confidence" || fieldName.startsWith("prod_prob_")) {
    return escapeHtml(formatPercentage(value));
  }

  if (fieldName.endsWith("_min")) {
    return escapeHtml(formatMinuteValue(value));
  }

  if (fieldName === "pair_next_day_flag") {
    return escapeHtml(Number(value) === 1 ? "Yes" : "No");
  }

  if (fieldName === "attendance_day" || isIsoDateLike(value)) {
    return escapeHtml(formatDateValue(value));
  }

  if (["best_shift_candidate", "prod_model_pred_shift", "final_shift", "working_shift_hint"].includes(fieldName)) {
    return `<span class="cell-code">${escapeHtml(String(value))}</span>`;
  }

  if (typeof value === "string" && value.length > 84) {
    const text = escapeHtml(value);
    return `<span class="cell-truncate" title="${text}">${text}</span>`;
  }

  return escapeHtml(String(value));
}

function isDerivedColumn(column) {
  return ["punch_in_time", "punch_out_time"].includes(column);
}

function splitPunches(pairPreview) {
  if (!pairPreview) {
    return [];
  }

  return String(pairPreview)
    .split("|")
    .map((part) => part.trim())
    .filter(Boolean);
}

function normalizeBooleanFlag(value) {
  if (typeof value === "boolean") {
    return value;
  }

  if (typeof value === "number") {
    return value === 1;
  }

  if (typeof value === "string") {
    const normalizedValue = value.trim().toLowerCase();
    return ["true", "1", "yes", "y"].includes(normalizedValue);
  }

  return false;
}

function hasRealCandidateShift(value) {
  if (value === null || value === undefined) {
    return false;
  }

  const normalizedValue = String(value).trim().toLowerCase();
  return normalizedValue !== "" && !["nan", "none", "null", "unknown"].includes(normalizedValue);
}

function computeValidPairFound(row) {
  if (row && "valid_reader_pair_found" in row) {
    return normalizeBooleanFlag(row.valid_reader_pair_found);
  }
  if (row && "reader_pair_found_flag" in row) {
    return normalizeBooleanFlag(row.reader_pair_found_flag);
  }
  const pairPunchCount = Number(row?.pair_punch_count);
  return Number.isFinite(pairPunchCount) && pairPunchCount >= 2 && hasRealCandidateShift(row?.best_shift_candidate);
}

function isValidPairFound(row) {
  if (row && ("valid_reader_pair_found" in row || "reader_pair_found_flag" in row || "pair_punch_count" in row || "best_shift_candidate" in row)) {
    return computeValidPairFound(row);
  }

  return normalizeBooleanFlag(row?.valid_pair_found || row?.valid_reader_pair_found);
}

function renderValidPairBadge(row) {
  const validPairFound = isValidPairFound(row);
  const label = validPairFound ? "Yes" : "No";
  const toneClass = validPairFound ? "pair-validity-yes" : "pair-validity-no";
  return `<span class="pair-validity ${toneClass}">${label}</span>`;
}

function renderCandidatePreview(row) {
  const candidateText = escapeHtml(String(row?.pair_preview || ""));
  const validPairFound = isValidPairFound(row);

  if (!validPairFound && candidateText) {
    return `
      <div class="candidate-preview candidate-preview-invalid">
        <span class="candidate-preview-state">No valid reader pair</span>
        <span class="candidate-preview-text" title="${candidateText}">${candidateText}</span>
      </div>
    `;
  }

  if (!validPairFound) {
    return '<span class="cell-text-muted">No valid reader pair</span>';
  }

  return `<span class="cell-preview" title="${candidateText}">${candidateText}</span>`;
}

function formatPreviewBlock(value) {
  if (value === null || value === undefined || value === "") {
    return '<span class="cell-text-muted">—</span>';
  }

  return escapeHtml(String(value));
}

function formatCandidateDetailBlock(row) {
  const candidateText = row?.pair_preview;
  if (!candidateText) {
    return isValidPairFound(row)
      ? '<span class="cell-text-muted">—</span>'
      : '<span class="cell-text-muted">No valid reader pair</span>';
  }

  if (!isValidPairFound(row)) {
    return `
      <div class="candidate-preview candidate-preview-invalid">
        <span class="candidate-preview-state">No valid reader pair</span>
        <span class="candidate-preview-text">${escapeHtml(String(candidateText))}</span>
      </div>
    `;
  }

  return escapeHtml(String(candidateText));
}

function normalizePredictionRows(rows) {
  return rows.map((row) => {
    const validPairFound = computeValidPairFound(row);
    const punches = splitPunches(row?.pair_preview);
    const fallbackPunchIn = validPairFound && row?.pair_start_min !== undefined && row?.pair_start_min !== null
      ? formatMinuteValue(row.pair_start_min)
      : "";
    const fallbackPunchOut = validPairFound && row?.pair_end_min !== undefined && row?.pair_end_min !== null
      ? formatMinuteValue(row.pair_end_min)
      : "";

    return {
      ...row,
      valid_pair_found: validPairFound,
      valid_reader_pair_found: "valid_reader_pair_found" in row ? row.valid_reader_pair_found : validPairFound,
      punch_in_time: validPairFound ? (row?.punch_in_time || punches[0] || fallbackPunchIn) : "",
      punch_out_time: validPairFound ? (row?.punch_out_time || punches[punches.length - 1] || fallbackPunchOut) : "",
    };
  });
}

function buildAvailableColumns(sourceColumns, rows) {
  const baseColumns = Array.isArray(sourceColumns) ? [...sourceColumns] : [];
  const firstRow = rows[0] || {};

  Object.keys(firstRow).forEach((column) => {
    if (!baseColumns.includes(column)) {
      baseColumns.push(column);
    }
  });

  return baseColumns;
}

function formatPercentage(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return String(value);
  }

  return `${(numericValue * 100).toFixed(2)}%`;
}

function formatMinuteValue(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return String(value);
  }

  return `${numericValue.toFixed(1)} min`;
}

function statusPill(value) {
  const status = String(value);
  return `<span class="status-pill ${statusToneClass(status)}">${escapeHtml(formatStatusLabel(status))}</span>`;
}

function statusChip(value) {
  const label = String(value);
  return `<span class="status-chip ${statusToneClass(label)}">${escapeHtml(formatStatusLabel(label))}</span>`;
}

function formatStatusLabel(value) {
  const normalizedValue = String(value).toUpperCase();
  if (normalizedValue === "WO") {
    return "Weekend Day Off";
  }
  if (normalizedValue === "NOT_APPLICABLE") {
    return "Not Applicable";
  }

  return String(value);
}

function statusToneClass(value) {
  const normalizedValue = String(value).toUpperCase();
  if (normalizedValue.includes("PREDICTED") || normalizedValue === "WORKING" || normalizedValue === "PREDICTED" || normalizedValue === "MATCH") {
    return "status-good";
  }
  if (normalizedValue === "MISMATCH") {
    return "status-withheld";
  }
  if (normalizedValue.includes("REVIEW")) {
    return "status-review";
  }
  if (normalizedValue.includes("WITHHELD")) {
    return "status-withheld";
  }
  if (normalizedValue.includes("ABSENT")) {
    return "status-withheld";
  }
  if (normalizedValue === "WO") {
    return "status-wo";
  }
  return "status-neutral";
}

function formatDateValue(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value || "");
  }

  return new Intl.DateTimeFormat("en-IN", {
    year: "numeric",
    month: "short",
    day: "2-digit",
  }).format(date);
}

function formatLocalInputDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function normalizeManualDateTimeInput(value) {
  return String(value || "").trim().replace("T", " ");
}

function isIsoDateLike(value) {
  return typeof value === "string" && /^\d{4}-\d{2}-\d{2}(T.*)?$/.test(value);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
