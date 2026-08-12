// ---------- pestañas ----------
const tabButtons = document.querySelectorAll(".tab-btn");
const panels = document.querySelectorAll(".panel");
tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabButtons.forEach((b) => b.classList.remove("active"));
    panels.forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// ---------- controles de transporte ----------
const post = (url, body) =>
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });

let currentPosition = 1;
let currentSide = "white";

document.getElementById("btn-start").addEventListener("click", () => post("/api/control/start"));
document.getElementById("btn-pause").addEventListener("click", () => post("/api/control/pause"));
document.getElementById("btn-reset").addEventListener("click", () =>
  post("/api/control/reset", {
    start_position: currentPosition,
    side_to_move: currentSide,
  })
);

const speedInput = document.getElementById("speed");
const speedValue = document.getElementById("speed-value");
speedInput.addEventListener("input", () => {
  speedValue.textContent = `${speedInput.value} pasos/s`;
  post("/api/control/speed", { value: Number(speedInput.value) });
});

const cfgPosition = document.getElementById("cfg-position");
const cfgPositionValue = document.getElementById("cfg-position-value");
const cfgSide = document.getElementById("cfg-side");
cfgPosition.addEventListener("input", () => (cfgPositionValue.textContent = cfgPosition.value));
document.getElementById("btn-apply-config").addEventListener("click", () => {
  currentPosition = Number(cfgPosition.value);
  currentSide = cfgSide.value;
  post("/api/control/reset", {
    start_position: currentPosition,
    side_to_move: currentSide,
  });
});

// ---------- render del tablero KRK ----------
const boardEl = document.getElementById("chess-board");
let lastFen = null;

function drawEnvironment(env) {
  if (!env || !env.svg) return;
  // Evita reescribir el DOM si la posición no cambió.
  if (env.fen === lastFen) return;
  lastFen = env.fen;
  boardEl.innerHTML = env.svg;

  const sideLabel = env.side_to_move === "white" ? "blancas" : "negras";
  document.getElementById("env-dims").textContent =
    `posición ${env.start_position} · ${sideLabel}`;

  let hint = "en juego";
  if (env.is_checkmate) hint = "jaque mate";
  else if (env.is_stalemate) hint = "ahogado";
  else if (env.is_check) hint = "jaque";
  else if (env.is_terminal) hint = "terminal";
  document.getElementById("env-status-hint").textContent = hint;
}

// ---------- mapa de calor Q ----------
const qCanvas = document.getElementById("qheatmap");
const qCtx = qCanvas.getContext("2d");

function drawHeatmap(grid) {
  const size = grid.length;
  const cell = qCanvas.width / size;
  let max = -Infinity, min = Infinity;
  grid.forEach((row) => row.forEach((v) => { if (v > max) max = v; if (v < min) min = v; }));
  const range = max - min || 1;

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const t = (grid[r][c] - min) / range;
      const g = Math.round(40 + t * 180);
      const b = Math.round(60 + (1 - t) * 120);
      qCtx.fillStyle = `rgb(20, ${g}, ${b})`;
      qCtx.fillRect(c * cell, r * cell, cell, cell);
    }
  }
}

// ---------- gráficas Chart.js ----------
const chartOpts = (yLabel) => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { display: false },
    y: {
      ticks: { color: "#6d7f91", font: { family: "IBM Plex Mono", size: 10 } },
      grid: { color: "#182129" },
      title: { display: false, text: yLabel },
    },
  },
  elements: { point: { radius: 0 }, line: { borderWidth: 2, tension: 0.25 } },
});

const rewardChart = new Chart(document.getElementById("chart-reward"), {
  type: "line",
  data: { labels: [], datasets: [{ data: [], borderColor: "#49f2a6" }] },
  options: chartOpts("recompensa"),
});
const stepsChart = new Chart(document.getElementById("chart-steps"), {
  type: "line",
  data: { labels: [], datasets: [{ data: [], borderColor: "#57bfff" }] },
  options: chartOpts("pasos"),
});
const epsilonChart = new Chart(document.getElementById("chart-epsilon"), {
  type: "line",
  data: { labels: [], datasets: [{ data: [], borderColor: "#ffb454" }] },
  options: chartOpts("epsilon"),
});

function updateChart(chart, values) {
  chart.data.labels = values.map((_, i) => i + 1);
  chart.data.datasets[0].data = values;
  chart.update("none");
}

// ---------- sondeo del estado ----------
async function poll() {
  try {
    const res = await fetch("/api/state");
    const state = await res.json();
    render(state);
  } catch (e) {
    // reintenta silenciosamente en el siguiente ciclo
  }
  setTimeout(poll, 200);
}

function render(state) {
  const env = state.environment;
  const agent = state.agent;

  const pill = document.getElementById("status-pill");
  const statusText = document.getElementById("status-text");
  if (state.running) {
    pill.classList.add("running");
    statusText.textContent = "entrenando";
  } else {
    pill.classList.remove("running");
    statusText.textContent = "en pausa";
  }
  speedInput.value = state.speed;
  speedValue.textContent = `${state.speed} pasos/s`;

  document.getElementById("ro-episode").textContent = agent.episode;
  document.getElementById("ro-steps").textContent = `${env.steps} / ${env.max_steps}`;
  document.getElementById("ro-ep-reward").textContent = agent.current_episode_reward.toFixed(2);
  document.getElementById("ro-action").textContent = agent.last_action || "—";
  document.getElementById("ro-turn").textContent =
    env.turn === "white" ? "blancas" : "negras";
  document.getElementById("ro-mode").textContent = agent.algo
    ? agent.algo
    : agent.last_exploration
      ? "exploración"
      : "explotación";
  drawEnvironment(env);

  if (cfgPosition.value !== String(env.start_position)) {
    cfgPosition.value = env.start_position;
    cfgPositionValue.textContent = env.start_position;
  }
  if (cfgSide.value !== env.side_to_move) {
    cfgSide.value = env.side_to_move;
  }
  currentPosition = env.start_position;
  currentSide = env.side_to_move;

  document.getElementById("st-episode").textContent = agent.episode;
  document.getElementById("st-epsilon").textContent =
    agent.epsilon < 1 ? agent.epsilon.toFixed(6) : agent.epsilon.toFixed(3);
  document.getElementById("st-avg-reward").textContent = agent.avg_reward_last_50.toFixed(2);
  document.getElementById("st-best-reward").textContent = agent.best_episode_reward.toFixed(2);
  document.getElementById("st-td-error").textContent = agent.td_error_ema.toFixed(4);
  document.getElementById("st-total-steps").textContent = agent.total_steps;

  updateChart(rewardChart, agent.reward_history);
  updateChart(stepsChart, agent.steps_history);
  updateChart(epsilonChart, agent.epsilon_history);
  drawHeatmap(agent.value_grid);
}

poll();
