const storageKeys = {
  trace: "paper-agent-trace-settings",
  config: "paper-agent-provider-config",
};

const defaultTraceSettings = {
  arxivUrl: "https://arxiv.org/abs/2305.10601",
  maxRounds: 3,
  topK: 3,
};

const defaultConfig = {
  protocol: "openai",
  modelId: "gpt-4o",
  baseUrl: "https://api.openai.com/v1",
  apiKey: "",
  outputsDir: "outputs",
  dataDir: "data",
  requestTimeout: 120,
  enableRateLimit: true,
};

const GRAPH_WIDTH = 920;
const GRAPH_HEIGHT = 460;
const WATCH_INTERVAL_MS = 3000;

const graphCanvas = document.getElementById("graph-canvas");
const graphTooltip = document.getElementById("graph-tooltip");
const graphCaptionText = document.getElementById("graph-caption-text");
const logList = document.getElementById("log-list");
const logCount = document.getElementById("log-count");
const metricRound = document.getElementById("metric-round");
const metricAnalyzed = document.getElementById("metric-analyzed");
const metricFrontier = document.getElementById("metric-frontier");
const metricDensity = document.getElementById("metric-density");
const statusCard = document.getElementById("status-card");
const traceForm = document.getElementById("trace-form");
const configForm = document.getElementById("config-form");
const replayButton = document.getElementById("replay-demo");
const resetGraphViewButton = document.getElementById("reset-graph-view");
const primarySubmitButton = traceForm.querySelector('.primary-button');

let watchTimer = null;
let logTimer = null;
let visibleGraph = { nodes: [], edges: [] };
let hoveredNodeId = null;
let graphViewport = null;
let currentRun = null;
let processLogEntries = [];
let currentSnapshotMeta = {
  fileName: "",
  snapshotKey: "",
  updatedAt: "",
};
let currentSnapshotBinding = {
  rootId: compactPaperId(defaultTraceSettings.arxivUrl),
  snapshotPath: buildSnapshotPath(defaultConfig.dataDir, compactPaperId(defaultTraceSettings.arxivUrl)),
};

const graphView = {
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  minScale: 0.55,
  maxScale: 2.6,
  isDragging: false,
  dragPointerId: null,
  lastClientX: 0,
  lastClientY: 0,
};

initializeTabs();
hydrateForms();
initializeGraphInteractions();
renderEmptyState();
startWatching(readTraceSettings());

traceForm.addEventListener("submit", (event) => {
  event.preventDefault();

  const formData = new FormData(traceForm);
  const traceSettings = {
    arxivUrl: String(formData.get("arxivUrl") || "").trim(),
    maxRounds: clampNumber(formData.get("maxRounds"), 1, 8, 3),
    topK: clampNumber(formData.get("topK"), 1, 8, 3),
  };

  localStorage.setItem(storageKeys.trace, JSON.stringify(traceSettings));
  startBackendRun(traceSettings);
});

configForm.addEventListener("submit", (event) => {
  event.preventDefault();

  const formData = new FormData(configForm);
  const config = {
    protocol: String(formData.get("protocol") || defaultConfig.protocol),
    modelId: String(formData.get("modelId") || "").trim(),
    baseUrl: String(formData.get("baseUrl") || "").trim(),
    apiKey: String(formData.get("apiKey") || ""),
    outputsDir: String(formData.get("outputsDir") || "").trim(),
    dataDir: String(formData.get("dataDir") || "").trim(),
    requestTimeout: clampNumber(formData.get("requestTimeout"), 10, 600, 120),
    enableRateLimit: Boolean(formData.get("enableRateLimit")),
  };

  localStorage.setItem(storageKeys.config, JSON.stringify(config));
  updateStatusCard(
    "Local config updated",
    "Provider settings are saved in browser storage for this prototype workspace.",
  );
});

replayButton.addEventListener("click", () => {
  startWatching(readTraceSettings(), { forceRefresh: true });
  if (currentRun?.runId) {
    startLogPolling(currentRun.runId, true);
  }
});

resetGraphViewButton.addEventListener("click", () => {
  resetGraphView();
  fitGraphToView();
  updateGraphViewport();
  syncTooltipPosition();
});

function initializeTabs() {
  const tabButtons = document.querySelectorAll(".tab-button");
  const panels = document.querySelectorAll(".tab-panel");

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.tabTarget;
      tabButtons.forEach((item) => item.classList.toggle("is-active", item === button));
      panels.forEach((panel) => panel.classList.toggle("is-active", panel.id === target));
    });
  });
}

function hydrateForms() {
  const traceSettings = readTraceSettings();
  const config = readConfig();

  document.getElementById("arxiv-url").value = traceSettings.arxivUrl;
  document.getElementById("max-rounds").value = traceSettings.maxRounds;
  document.getElementById("top-k").value = traceSettings.topK;

  document.getElementById("protocol").value = config.protocol;
  document.getElementById("model-id").value = config.modelId;
  document.getElementById("base-url").value = config.baseUrl;
  document.getElementById("api-key").value = config.apiKey;
  document.getElementById("outputs-dir").value = config.outputsDir;
  document.getElementById("data-dir").value = config.dataDir;
  document.getElementById("request-timeout").value = config.requestTimeout;
  document.getElementById("enable-rate-limit").checked = config.enableRateLimit;
}

function readTraceSettings() {
  return readStoredJSON(storageKeys.trace, defaultTraceSettings);
}

function readConfig() {
  return readStoredJSON(storageKeys.config, defaultConfig);
}

function readStoredJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) {
      return { ...fallback };
    }
    return { ...fallback, ...JSON.parse(raw) };
  } catch {
    return { ...fallback };
  }
}

function renderEmptyState() {
  visibleGraph = { nodes: [], edges: [] };
  currentSnapshotMeta = {
    fileName: "",
    snapshotKey: "",
    updatedAt: "",
  };
  metricRound.textContent = `0 / ${defaultTraceSettings.maxRounds}`;
  metricAnalyzed.textContent = "0";
  metricFrontier.textContent = "0";
  metricDensity.textContent = "0 edges";
  graphCaptionText.textContent =
    "Enter an arXiv paper and the UI will poll the matching graph snapshot under the repository data directory.";
  updateStatusCard(
    "Awaiting a root paper",
    "The observatory will switch to the matching graph snapshot once a readable JSON file appears.",
  );
  processLogEntries = [];
  renderLogs([]);
  renderGraph([], []);
}

async function startBackendRun(traceSettings) {
  const config = readConfig();
  setPrimaryActionState(true, "Starting...");
  updateStatusCard(
    "Starting trace",
    "The UI is launching the backend trace process and will attach graph and log monitors when the run starts.",
  );

  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        ...traceSettings,
        config,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }

    currentRun = {
      runId: payload.run_id,
      rootId: payload.root_id,
      snapshotPath: payload.snapshot_path,
      logPath: payload.log_path,
    };
    currentSnapshotBinding = {
      rootId: payload.root_id,
      snapshotPath: payload.snapshot_path,
    };
    processLogEntries = [];

    startWatching(traceSettings, { resetView: true, forceRefresh: true });
    startLogPolling(currentRun.runId, true);
    setPrimaryActionState(true, "Search Running");
    updateStatusCard(
      "Trace launched",
      `${payload.root_id} is now running in the backend process. Graph snapshots and trace logs will refresh automatically.`,
    );
  } catch (error) {
    currentRun = null;
    updateStatusCard(
      "Trace start failed",
      `The backend process could not be started: ${error.message}`,
    );
    processLogEntries = [
      buildLogEntry("error", `Failed to start backend trace: ${error.message}.`),
      buildLogEntry("hint", "Start the UI with `python3 dev_server.py` so the API endpoints are available."),
    ];
    renderLogs(processLogEntries);
    setPrimaryActionState(false, "Start Search");
  }
}

function setPrimaryActionState(disabled, label) {
  primarySubmitButton.disabled = disabled;
  primarySubmitButton.textContent = label;
}

function startWatching(traceSettings = readTraceSettings(), options = {}) {
  const normalized = {
    ...traceSettings,
    arxivUrl: traceSettings.arxivUrl || defaultTraceSettings.arxivUrl,
    maxRounds: clampNumber(traceSettings.maxRounds, 1, 8, 3),
    topK: clampNumber(traceSettings.topK, 1, 8, 3),
  };

  if (watchTimer) {
    clearInterval(watchTimer);
  }

  if (options.resetView) {
    resetGraphView();
  }

  if (!currentRun?.snapshotPath) {
    const rootId = compactPaperId(normalized.arxivUrl);
    currentSnapshotBinding = {
      rootId,
      snapshotPath: buildSnapshotPath(readConfig().dataDir, rootId),
    };
  }

  pollGraphSnapshot(normalized, Boolean(options.forceRefresh));
  watchTimer = setInterval(() => {
    pollGraphSnapshot(normalized, false);
  }, WATCH_INTERVAL_MS);
}

function startLogPolling(runId, forceRefresh) {
  if (logTimer) {
    clearInterval(logTimer);
  }
  pollRunLogs(runId, forceRefresh ? 0 : currentRun?.cursor || 0, true);
  logTimer = setInterval(() => {
    pollRunLogs(runId, currentRun?.cursor || 0, false);
  }, 2000);
}

async function pollRunLogs(runId, cursor, reset) {
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/logs?cursor=${cursor}`, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }

    currentRun = {
      ...(currentRun || {}),
      runId: payload.run_id,
      rootId: payload.root_id,
      status: payload.status,
      cursor: payload.next_cursor,
      snapshotPath: payload.snapshot_path,
      logPath: payload.log_path,
    };
    currentSnapshotBinding = {
      rootId: payload.root_id,
      snapshotPath: payload.snapshot_path,
    };

    const nextEntries = payload.lines.map((line) => buildLogEntry(`log ${line.cursor}`, line.text));
    if (reset) {
      processLogEntries = nextEntries;
      renderLogs(processLogEntries);
    } else if (nextEntries.length) {
      prependProcessLogEntries(nextEntries);
    }

    if (payload.status === "completed" || payload.status === "failed") {
      clearInterval(logTimer);
      logTimer = null;
      currentRun = {
        ...(currentRun || {}),
        status: payload.status,
      };
      setPrimaryActionState(false, "Start Search");
    }
  } catch (error) {
    prependProcessLogEntries([buildLogEntry("error", `Log polling failed: ${error.message}.`)]);
  }
}

function prependProcessLogEntries(entries) {
  processLogEntries = [...entries, ...processLogEntries].slice(0, 200);
  renderLogs(processLogEntries);
}

async function pollGraphSnapshot(traceSettings, forceRefresh) {
  const rootId = currentSnapshotBinding.rootId || compactPaperId(traceSettings.arxivUrl);
  const snapshotPath = currentSnapshotBinding.snapshotPath || buildSnapshotPath(readConfig().dataDir, rootId);
  const snapshotUrl = buildFetchUrl(snapshotPath, forceRefresh ? Date.now() : null);

  updateStatusCard(
    "Watching graph snapshot",
    `${rootId} is bound to ${snapshotPath}. The UI is polling this JSON file in read-only mode.`,
  );

  try {
    const response = await fetch(snapshotUrl, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const snapshot = await response.json();
    applySnapshot(snapshot, traceSettings, rootId, snapshotPath, response.headers.get("last-modified") || "", forceRefresh);
  } catch (error) {
    renderSnapshotMissingState(rootId, snapshotPath, error);
  }
}

function applySnapshot(snapshot, traceSettings, rootId, snapshotPath, updatedAt, forceRefresh) {
  const prepared = prepareGraph(snapshot, rootId);
  const snapshotKey = JSON.stringify({
    nodes: prepared.nodes.map((node) => [node.id, node.status, node.rawStatus, node.round, node.title]).sort(compareTuple),
    edges: prepared.edges.map((edge) => [edge.source, edge.target]).sort(compareTuple),
    meta: [
      prepared.meta.status,
      prepared.meta.phase,
      prepared.meta.detail,
      prepared.meta.currentRound,
      prepared.meta.maxRounds,
      prepared.meta.frontierSize,
      prepared.meta.updatedAt,
      prepared.meta.reportPath,
    ],
  });
  const hasChanged = snapshotKey !== currentSnapshotMeta.snapshotKey;

  currentSnapshotMeta = {
    fileName: snapshotPath,
    snapshotKey,
    updatedAt,
  };

  visibleGraph = {
    nodes: cloneNodes(prepared.nodes),
    edges: cloneEdges(prepared.edges),
  };

  if (hasChanged || forceRefresh) {
    renderGraph(visibleGraph.nodes, visibleGraph.edges);
    fitGraphToView();
    updateGraphViewport();
  }

  renderSnapshotSummary(traceSettings, rootId, snapshotPath, prepared, hasChanged, updatedAt);
}

function renderSnapshotSummary(traceSettings, rootId, snapshotPath, prepared, hasChanged, updatedAt) {
  const analyzed = prepared.nodes.filter(isAnalyzedNode).length;
  const frontier = prepared.meta.frontierSize ?? prepared.nodes.filter((node) => node.status === "pending").length;
  const maxRoundSeen = Math.max(
    prepared.meta.currentRound ?? 0,
    prepared.nodes.reduce((maxRound, node) => Math.max(maxRound, node.round), 0),
  );
  const totalRounds = prepared.meta.maxRounds ?? traceSettings.maxRounds;
  const failed = prepared.nodes.filter((node) => node.status === "failed").length;

  metricRound.textContent = `${maxRoundSeen} / ${totalRounds}`;
  metricAnalyzed.textContent = String(analyzed);
  metricFrontier.textContent = String(frontier);
  metricDensity.textContent = `${prepared.edges.length} edges`;

  const effectiveUpdatedAt = prepared.meta.updatedAtLabel || (updatedAt ? new Date(updatedAt).toLocaleString("zh-CN") : "");
  const freshness = effectiveUpdatedAt ? `Last updated ${effectiveUpdatedAt}.` : "Polling for fresh file changes.";
  const phaseLabel = formatPhaseLabel(prepared.meta.phase);
  graphCaptionText.textContent = hasChanged
    ? `Snapshot ${snapshotPath} changed. ${phaseLabel} ${prepared.meta.detail || `${prepared.nodes.length} paper nodes are now visible in the lineage graph.`}`
    : `Snapshot ${snapshotPath} is unchanged. ${freshness}`;

  updateStatusCard(
    hasChanged ? formatSnapshotStatus(prepared.meta.status, true) : formatSnapshotStatus(prepared.meta.status, false),
    `${rootId} currently has ${prepared.nodes.length} node(s), ${prepared.edges.length} edge(s), ${analyzed} analyzed paper(s), ${frontier} frontier paper(s), and ${failed} failed paper(s). ${phaseLabel} ${prepared.meta.detail || ""} ${freshness}`.trim(),
  );

}

function renderSnapshotMissingState(rootId, snapshotPath, error) {
  visibleGraph = { nodes: [], edges: [] };
  currentSnapshotMeta = {
    fileName: snapshotPath,
    snapshotKey: "",
    updatedAt: "",
  };
  metricRound.textContent = `0 / ${readTraceSettings().maxRounds}`;
  metricAnalyzed.textContent = "0";
  metricFrontier.textContent = "0";
  metricDensity.textContent = "0 edges";
  graphCaptionText.textContent =
    "No readable graph snapshot is available yet. Run the backend from the repo root and let it write JSON into the configured data directory.";
  updateStatusCard(
    "Snapshot not available",
    `Tried to read ${snapshotPath} for ${rootId}, but the browser could not fetch it (${error.message}). Serve the repository root so both ui and data are reachable.`,
  );
  if (!currentRun?.runId) {
    renderLogs([
      buildLogEntry("watcher", `Polling ${snapshotPath} for ${rootId}.`),
      buildLogEntry("error", `Snapshot fetch failed: ${error.message}.`),
      buildLogEntry("hint", "Use `python3 dev_server.py` from the repository root, then open `/ui/`."),
    ]);
  } else if (!processLogEntries.length) {
    renderLogs([
      buildLogEntry("watcher", `Waiting for backend logs while polling ${snapshotPath}.`),
      buildLogEntry("error", `Snapshot fetch failed: ${error.message}.`),
    ]);
  }
  renderGraph([], []);
}

function prepareGraph(snapshot, rootId) {
  const rawNodes = Array.isArray(snapshot?.nodes) ? snapshot.nodes : [];
  const rawEdges = Array.isArray(snapshot?.links) ? snapshot.links : [];
  const meta = normalizeSnapshotMeta(snapshot?.meta);
  const nodes = rawNodes.map((node) => ({
    id: String(node.id || node.arxiv_id || "unknown"),
    label: String(node.arxiv_id || node.id || "unknown"),
    title: String(node.title || node.arxiv_id || node.id || "Untitled paper"),
    url: buildPaperUrlFromNode(node),
    round: clampNumber(node.round_added, 0, 99, 0),
    status: normalizeStatus(node.status, node.arxiv_id === rootId),
    rawStatus: String(node.status || "").toLowerCase(),
    authors: Array.isArray(node.authors) ? node.authors : [],
    year: node.year ?? null,
    abstract: String(node.abstract || ""),
    method: String(node.method || ""),
    problemSolved: String(node.problem_solved || ""),
    x: 0,
    y: 0,
  }));

  const edges = rawEdges
    .map((edge) => ({
      source: String(edge.source || ""),
      target: String(edge.target || ""),
    }))
    .filter((edge) => edge.source && edge.target);

  applyLayout(nodes, edges, rootId);
  return { nodes, edges, meta };
}

function normalizeSnapshotMeta(meta) {
  const updatedAt = Number(meta?.updated_at);
  return {
    rootId: String(meta?.root_id || ""),
    status: String(meta?.status || "running"),
    phase: String(meta?.phase || "snapshot_loaded"),
    detail: String(meta?.detail || "").trim(),
    currentRound: Number.isFinite(Number(meta?.current_round)) ? Number(meta.current_round) : null,
    maxRounds: Number.isFinite(Number(meta?.max_rounds)) ? Number(meta.max_rounds) : null,
    frontier: Array.isArray(meta?.frontier) ? meta.frontier.map((item) => String(item)) : [],
    frontierSize: Number.isFinite(Number(meta?.frontier_size)) ? Number(meta.frontier_size) : null,
    skippedRefs: Array.isArray(meta?.skipped_refs) ? meta.skipped_refs.map((item) => String(item)) : [],
    lineageChain: Array.isArray(meta?.lineage_chain) ? meta.lineage_chain.map((item) => String(item)) : [],
    lineageRationale: String(meta?.lineage_rationale || "").trim(),
    reportPath: String(meta?.report_path || "").trim(),
    updatedAt: Number.isFinite(updatedAt) ? updatedAt : null,
    updatedAtLabel: Number.isFinite(updatedAt) ? new Date(updatedAt * 1000).toLocaleString("zh-CN") : "",
  };
}

function applyLayout(nodes, edges, rootId) {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const childrenByParent = new Map();
  const indegree = new Map(nodes.map((node) => [node.id, 0]));

  edges.forEach((edge) => {
    if (!childrenByParent.has(edge.source)) {
      childrenByParent.set(edge.source, []);
    }
    childrenByParent.get(edge.source).push(edge.target);
    indegree.set(edge.target, (indegree.get(edge.target) || 0) + 1);
  });

  const rootNode =
    nodeMap.get(rootId) ||
    nodes.find((node) => node.status === "root") ||
    nodes.find((node) => (indegree.get(node.id) || 0) === 0) ||
    nodes[0];

  const levels = new Map();
  if (rootNode) {
    assignLevels(rootNode.id, 0, childrenByParent, levels, new Set());
  }

  nodes.forEach((node) => {
    if (!levels.has(node.id)) {
      levels.set(node.id, Math.max(node.round, 0));
    }
  });

  const levelBuckets = new Map();
  nodes.forEach((node) => {
    const level = levels.get(node.id) || 0;
    if (!levelBuckets.has(level)) {
      levelBuckets.set(level, []);
    }
    levelBuckets.get(level).push(node);
  });

  const sortedLevels = [...levelBuckets.keys()].sort((left, right) => left - right);
  const verticalGap = sortedLevels.length > 1 ? (GRAPH_HEIGHT - 120) / (sortedLevels.length - 1) : 0;

  sortedLevels.forEach((level) => {
    const bucket = levelBuckets.get(level);
    bucket.sort((left, right) => compareNodeOrder(left, right));
    const horizontalGap = GRAPH_WIDTH / (bucket.length + 1);
    bucket.forEach((node, index) => {
      node.x = horizontalGap * (index + 1);
      node.y = 72 + level * verticalGap;
    });
  });
}

function assignLevels(nodeId, level, childrenByParent, levels, stack) {
  if (stack.has(nodeId)) {
    return;
  }
  const nextLevel = levels.has(nodeId) ? Math.min(levels.get(nodeId), level) : level;
  levels.set(nodeId, nextLevel);
  stack.add(nodeId);
  const children = childrenByParent.get(nodeId) || [];
  children.forEach((childId) => assignLevels(childId, nextLevel + 1, childrenByParent, levels, stack));
  stack.delete(nodeId);
}

function compareNodeOrder(left, right) {
  if (left.status !== right.status) {
    return statusOrder(left.status) - statusOrder(right.status);
  }
  if (left.year !== right.year) {
    return (left.year || 9999) - (right.year || 9999);
  }
  return left.label.localeCompare(right.label);
}

function statusOrder(status) {
  const order = {
    root: 0,
    analyzed: 1,
    pending: 2,
    failed: 3,
    skipped: 4,
    unknown: 5,
  };
  return order[status] ?? order.unknown;
}

function normalizeStatus(status, isRoot) {
  if (isRoot) {
    return "root";
  }
  const value = String(status || "").toLowerCase();
  if (value === "analyzed" || value === "pending" || value === "failed" || value === "skipped") {
    return value;
  }
  return "pending";
}

function buildSnapshotPath(dataDir, rootId) {
  const cleanDir = String(dataDir || "data").replace(/^\.\//, "").replace(/\/+$/, "");
  return `${cleanDir || "data"}/${rootId.replaceAll("/", "_")}_graph.json`;
}

function buildFetchUrl(snapshotPath, cacheBustValue) {
  const relativePath = `../${snapshotPath}`;
  if (!cacheBustValue) {
    return relativePath;
  }
  return `${relativePath}?t=${cacheBustValue}`;
}

function buildPaperUrlFromNode(node) {
  const arxivId = String(node.arxiv_id || node.id || "");
  if (/(^\d{4}\.\d{4,5}(v\d+)?$)|(^[a-z\-]+\/\d{7}(v\d+)?$)/i.test(arxivId)) {
    return `https://arxiv.org/abs/${encodeURIComponent(arxivId)}`;
  }
  return "";
}

function buildLogEntry(time, text) {
  return { time, text };
}

function isAnalyzedNode(node) {
  return node.status === "analyzed" || (node.status === "root" && node.rawStatus === "analyzed");
}

function formatPhaseLabel(phase) {
  const labels = {
    ingest_completed: "Ingest completed.",
    round_started: "Round started.",
    metadata_fetched: "Metadata fetched.",
    paper_analyzed: "Paper analyzed.",
    paper_failed: "Paper failed.",
    candidate_added: "Candidate added.",
    edge_added: "Citation edge added.",
    round_completed: "Round completed.",
    evaluation_started: "Evaluation started.",
    evaluation_completed: "Evaluation completed.",
    evaluation_failed: "Evaluation failed.",
    report_completed: "Report completed.",
    snapshot_loaded: "Snapshot loaded.",
  };
  return labels[phase] || `Phase: ${phase}.`;
}

function formatSnapshotStatus(status, changed) {
  const normalized = String(status || "running").toLowerCase();
  if (normalized === "completed") {
    return changed ? "Trace completed" : "Trace synchronized";
  }
  if (normalized === "error") {
    return changed ? "Trace error updated" : "Trace error synchronized";
  }
  return changed ? "Trace updated" : "Trace synchronized";
}

function renderLogs(entries) {
  logCount.textContent = `${entries.length} entries`;
  logList.innerHTML = entries
    .map(
      (entry) => `
        <li class="log-item">
          <span class="log-item__time">${escapeHtml(entry.time)}</span>
          <p class="log-item__text">${escapeHtml(entry.text)}</p>
        </li>
      `,
    )
    .join("");
}

function updateStatusCard(title, text) {
  statusCard.innerHTML = `
    <span class="status-card__dot"></span>
    <div>
      <p class="status-card__title">${escapeHtml(title)}</p>
      <p class="status-card__text">${escapeHtml(text)}</p>
    </div>
  `;
}

function renderGraph(nodes, edges) {
  visibleGraph = {
    nodes: cloneNodes(nodes),
    edges: cloneEdges(edges),
  };

  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const edgeMarkup = edges
    .map((edge, index) => {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) {
        return "";
      }
      const midY = (source.y + target.y) / 2;
      return `
        <g class="edge" style="animation-delay:${index * 50}ms">
          <path d="M ${source.x} ${source.y} C ${source.x} ${midY}, ${target.x} ${midY}, ${target.x} ${target.y}" />
        </g>
      `;
    })
    .join("");

  const nodeMarkup = nodes
    .map(
      (node, index) => `
        <g
          class="node node--${escapeHtml(node.status)}${hoveredNodeId === node.id ? " is-hovered" : ""}"
          transform="translate(${node.x}, ${node.y})"
          style="animation-delay:${index * 70}ms"
          tabindex="0"
          role="link"
          data-node-id="${escapeHtml(node.id)}"
        >
          <circle r="${node.status === "root" ? 18 : 14}"></circle>
        </g>
      `,
    )
    .join("");

  graphCanvas.innerHTML = `<g id="graph-viewport" transform="translate(${graphView.offsetX} ${graphView.offsetY}) scale(${graphView.scale})">${edgeMarkup}${nodeMarkup}</g>`;
  graphViewport = document.getElementById("graph-viewport");
  syncTooltipPosition();
}

function fitGraphToView() {
  if (!visibleGraph.nodes.length) {
    return;
  }
  const bounds = visibleGraph.nodes.reduce(
    (accumulator, node) => ({
      minX: Math.min(accumulator.minX, node.x),
      maxX: Math.max(accumulator.maxX, node.x),
      minY: Math.min(accumulator.minY, node.y),
      maxY: Math.max(accumulator.maxY, node.y),
    }),
    { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity },
  );
  const width = Math.max(bounds.maxX - bounds.minX, 120);
  const height = Math.max(bounds.maxY - bounds.minY, 120);
  const paddedScale = Math.min((GRAPH_WIDTH - 120) / width, (GRAPH_HEIGHT - 120) / height, 1.2);
  graphView.scale = clampFloat(paddedScale, graphView.minScale, graphView.maxScale);
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;
  graphView.offsetX = GRAPH_WIDTH / 2 - centerX * graphView.scale;
  graphView.offsetY = GRAPH_HEIGHT / 2 - centerY * graphView.scale;
}

function updateGraphViewport() {
  if (!graphViewport) {
    return;
  }
  graphViewport.setAttribute(
    "transform",
    `translate(${graphView.offsetX} ${graphView.offsetY}) scale(${graphView.scale})`,
  );
}

function updateHoveredNode(nextHoveredNodeId) {
  hoveredNodeId = nextHoveredNodeId;
  const nodeElements = graphCanvas.querySelectorAll(".node");
  nodeElements.forEach((element) => {
    element.classList.toggle("is-hovered", element.dataset.nodeId === hoveredNodeId);
  });
}

function initializeGraphInteractions() {
  graphCanvas.addEventListener("pointerdown", onGraphPointerDown);
  graphCanvas.addEventListener("pointermove", onGraphPointerMove);
  graphCanvas.addEventListener("pointerup", onGraphPointerUp);
  graphCanvas.addEventListener("pointerleave", onGraphPointerLeave);
  graphCanvas.addEventListener("wheel", onGraphWheel, { passive: false });
  graphCanvas.addEventListener("mouseover", onGraphMouseOver);
  graphCanvas.addEventListener("mouseout", onGraphMouseOut);
  graphCanvas.addEventListener("click", onGraphClick);
  graphCanvas.addEventListener("keydown", onGraphKeyDown);
}

function onGraphPointerDown(event) {
  if (event.target.closest(".node")) {
    return;
  }
  graphView.isDragging = true;
  graphView.dragPointerId = event.pointerId;
  graphView.lastClientX = event.clientX;
  graphView.lastClientY = event.clientY;
  graphCanvas.classList.add("is-dragging");
  graphCanvas.setPointerCapture(event.pointerId);
  hideTooltip();
}

function onGraphPointerMove(event) {
  if (!graphView.isDragging || graphView.dragPointerId !== event.pointerId) {
    return;
  }
  const rect = graphCanvas.getBoundingClientRect();
  const dx = ((event.clientX - graphView.lastClientX) * GRAPH_WIDTH) / rect.width;
  const dy = ((event.clientY - graphView.lastClientY) * GRAPH_HEIGHT) / rect.height;
  graphView.offsetX += dx;
  graphView.offsetY += dy;
  graphView.lastClientX = event.clientX;
  graphView.lastClientY = event.clientY;
  updateGraphViewport();
  syncTooltipPosition();
}

function onGraphPointerUp(event) {
  if (graphView.dragPointerId !== event.pointerId) {
    return;
  }
  graphView.isDragging = false;
  graphView.dragPointerId = null;
  graphCanvas.classList.remove("is-dragging");
  if (graphCanvas.hasPointerCapture(event.pointerId)) {
    graphCanvas.releasePointerCapture(event.pointerId);
  }
}

function onGraphPointerLeave() {
  if (!graphView.isDragging) {
    updateHoveredNode(null);
    hideTooltip();
  }
}

function onGraphWheel(event) {
  event.preventDefault();
  const point = clientToSvgPoint(event.clientX, event.clientY);
  const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
  const nextScale = clampFloat(graphView.scale * zoomFactor, graphView.minScale, graphView.maxScale);
  const worldX = (point.x - graphView.offsetX) / graphView.scale;
  const worldY = (point.y - graphView.offsetY) / graphView.scale;
  graphView.offsetX = point.x - worldX * nextScale;
  graphView.offsetY = point.y - worldY * nextScale;
  graphView.scale = nextScale;
  updateGraphViewport();
  syncTooltipPosition();
}

function onGraphMouseOver(event) {
  const nodeElement = event.target.closest(".node");
  if (!nodeElement) {
    return;
  }
  updateHoveredNode(nodeElement.dataset.nodeId);
  showTooltip(hoveredNodeId);
}

function onGraphMouseOut(event) {
  const nodeElement = event.target.closest(".node");
  if (!nodeElement) {
    return;
  }
  const nextNode = event.relatedTarget?.closest?.(".node");
  if (nextNode && nextNode.dataset.nodeId === nodeElement.dataset.nodeId) {
    return;
  }
  updateHoveredNode(null);
  hideTooltip();
}

function onGraphClick(event) {
  const nodeElement = event.target.closest(".node");
  if (!nodeElement) {
    return;
  }
  const node = visibleGraph.nodes.find((item) => item.id === nodeElement.dataset.nodeId);
  if (node?.url) {
    window.open(node.url, "_blank", "noopener,noreferrer");
  }
}

function onGraphKeyDown(event) {
  const nodeElement = event.target.closest(".node");
  if (!nodeElement) {
    return;
  }
  if (event.key !== "Enter" && event.key !== " ") {
    return;
  }
  event.preventDefault();
  const node = visibleGraph.nodes.find((item) => item.id === nodeElement.dataset.nodeId);
  if (node?.url) {
    window.open(node.url, "_blank", "noopener,noreferrer");
  }
}

function showTooltip(nodeId) {
  const node = visibleGraph.nodes.find((item) => item.id === nodeId);
  if (!node) {
    hideTooltip();
    return;
  }
  const meta = [node.label];
  if (node.year) {
    meta.push(String(node.year));
  }
  if (node.status) {
    meta.push(node.status);
  }
  graphTooltip.innerHTML = `
    <strong>${escapeHtml(node.title)}</strong>
    <span class="graph-tooltip__meta">${escapeHtml(meta.join(" · "))}${node.url ? " · click to open paper" : ""}</span>
  `;
  graphTooltip.hidden = false;
  syncTooltipPosition();
}

function syncTooltipPosition() {
  if (graphTooltip.hidden || !hoveredNodeId) {
    return;
  }
  const node = visibleGraph.nodes.find((item) => item.id === hoveredNodeId);
  if (!node) {
    return;
  }
  const rect = graphCanvas.getBoundingClientRect();
  const screenX = (node.x * graphView.scale + graphView.offsetX) * (rect.width / GRAPH_WIDTH);
  const screenY = (node.y * graphView.scale + graphView.offsetY) * (rect.height / GRAPH_HEIGHT);
  graphTooltip.style.left = `${screenX}px`;
  graphTooltip.style.top = `${screenY}px`;
}

function hideTooltip() {
  graphTooltip.hidden = true;
}

function resetGraphView() {
  graphView.scale = 1;
  graphView.offsetX = 0;
  graphView.offsetY = 0;
  graphView.isDragging = false;
  graphView.dragPointerId = null;
  graphCanvas.classList.remove("is-dragging");
  updateHoveredNode(null);
  hideTooltip();
}

function clientToSvgPoint(clientX, clientY) {
  const rect = graphCanvas.getBoundingClientRect();
  return {
    x: ((clientX - rect.left) / rect.width) * GRAPH_WIDTH,
    y: ((clientY - rect.top) / rect.height) * GRAPH_HEIGHT,
  };
}

function compactPaperId(url) {
  const match = String(url).match(/(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+\/\d{7}(?:v\d+)?)/i);
  if (match) {
    return match[1];
  }
  return "paper.root";
}

function clampNumber(value, min, max, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, Math.round(parsed)));
}

function clampFloat(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function truncateText(value, limit) {
  const text = String(value || "").trim();
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, limit - 1)}...`;
}

function compareTuple(left, right) {
  return JSON.stringify(left).localeCompare(JSON.stringify(right));
}

function cloneNodes(nodes) {
  return nodes.map((node) => ({ ...node }));
}

function cloneEdges(edges) {
  return edges.map((edge) => ({ ...edge }));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
