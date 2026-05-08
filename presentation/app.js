const headlineByYear = [
  { year: 2008, headlineRows: 141, tradingDays: 106, avg: 1.33 },
  { year: 2009, headlineRows: 113, tradingDays: 90, avg: 1.26 },
  { year: 2010, headlineRows: 209, tradingDays: 132, avg: 1.58 },
  { year: 2011, headlineRows: 547, tradingDays: 216, avg: 2.53 },
  { year: 2012, headlineRows: 722, tradingDays: 241, avg: 3.0 },
  { year: 2013, headlineRows: 775, tradingDays: 241, avg: 3.22 },
  { year: 2014, headlineRows: 856, tradingDays: 242, avg: 3.54 },
  { year: 2015, headlineRows: 818, tradingDays: 243, avg: 3.37 },
  { year: 2016, headlineRows: 801, tradingDays: 240, avg: 3.34 },
  { year: 2017, headlineRows: 704, tradingDays: 228, avg: 3.09 },
  { year: 2018, headlineRows: 837, tradingDays: 243, avg: 3.44 },
  { year: 2019, headlineRows: 923, tradingDays: 248, avg: 3.72 },
  { year: 2020, headlineRows: 1515, tradingDays: 249, avg: 6.08 },
  { year: 2021, headlineRows: 1301, tradingDays: 250, avg: 5.2 },
  { year: 2022, headlineRows: 2316, tradingDays: 251, avg: 9.23 },
  { year: 2023, headlineRows: 4748, tradingDays: 244, avg: 19.46 },
  { year: 2024, headlineRows: 827, tradingDays: 43, avg: 19.23 }
];

const regimes = [
  {
    name: "2008-2009 financial crisis",
    days: 196,
    avgHeadlines: 1.3,
    upDayRate: 0.5408,
    avgVix: 33.0
  },
  {
    name: "2020 COVID shock",
    days: 249,
    avgHeadlines: 6.1,
    upDayRate: 0.5663,
    avgVix: 29.2
  },
  {
    name: "2022-2023 rate-hike cycle",
    days: 495,
    avgHeadlines: 14.3,
    upDayRate: 0.4869,
    avgVix: 21.3
  },
  {
    name: "Other",
    days: 2566,
    avgHeadlines: 3.6,
    upDayRate: 0.5518,
    avgVix: 16.9
  }
];

const terciles = [
  { label: "Low", avgReturn: 0.0003313, upDayRate: 0.5261 },
  { label: "Middle", avgReturn: 0.0005364, upDayRate: 0.5616 },
  { label: "High", avgReturn: 0.0004903, upDayRate: 0.5415 }
];

const modelMetrics = [
  { label: "Majority baseline", accuracy: 0.4953, balanced_accuracy: 0.5, auc: 0.5 },
  { label: "Logit all", accuracy: 0.4935, balanced_accuracy: 0.4981, auc: 0.5094 },
  { label: "XGB all", accuracy: 0.4953, balanced_accuracy: 0.5, auc: 0.522 },
  { label: "XGB VADER", accuracy: 0.5102, balanced_accuracy: 0.514, auc: 0.5071 },
  { label: "Logit FinBERT", accuracy: 0.4972, balanced_accuracy: 0.4986, auc: 0.5218 }
];

const backtests = [
  {
    label: "Buy & hold",
    strategy_cumulative_return: 0.0697,
    exposure: 1,
    strategy_max_drawdown: -0.2538,
    trades: 0,
    sharpe: 0.261
  },
  {
    label: "Logit all",
    strategy_cumulative_return: 0.0715,
    exposure: 0.994,
    strategy_max_drawdown: -0.2524,
    trades: 7,
    sharpe: 0.266
  },
  {
    label: "XGB all",
    strategy_cumulative_return: 0.0696,
    exposure: 1,
    strategy_max_drawdown: -0.2538,
    trades: 1,
    sharpe: 0.261
  },
  {
    label: "Logit FinBERT",
    strategy_cumulative_return: 0.1312,
    exposure: 0.648,
    strategy_max_drawdown: -0.152,
    trades: 134,
    sharpe: 0.453
  },
  {
    label: "XGB VADER",
    strategy_cumulative_return: 0.0913,
    exposure: 0.903,
    strategy_max_drawdown: -0.2234,
    trades: 87,
    sharpe: 0.32
  }
];

const rateHikeCards = [
  { label: "Logit all accuracy", value: "48.5%", note: "rate-hike test slice" },
  { label: "Logit all AUC", value: "0.499", note: "near random ranking" },
  { label: "XGB all accuracy", value: "48.7%", note: "same regime" },
  { label: "XGB all AUC", value: "0.513", note: "small ranking lift" }
];

const signalReality = [
  { label: "All-feature exposure", value: "99-100%", note: "Logit all and XGB all are almost always long" },
  { label: "FinBERT exposure", value: "64.8%", note: "more selective, higher selected test return" },
  { label: "VADER trades", value: "87", note: "more active than all-feature models" }
];

const palette = {
  blue: "#2f6f9f",
  green: "#3f7d5a",
  red: "#a94e43",
  gold: "#b07b2e",
  charcoal: "#27313a",
  muted: "#5f6f7f",
  line: "#d6dee7"
};

function pct(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function moneyPct(value, digits = 2) {
  return `${(value * 100).toFixed(digits)}%`;
}

function setActiveButton(buttons, activeButton) {
  buttons.forEach((button) => {
    button.classList.toggle("active", button === activeButton);
  });
}

function renderHeadlineChart() {
  const el = document.querySelector("#headlineChart");
  const width = 900;
  const height = 380;
  const margin = { top: 26, right: 24, bottom: 56, left: 58 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const max = Math.max(...headlineByYear.map((d) => d.headlineRows));
  const barGap = 8;
  const barW = (innerW - barGap * (headlineByYear.length - 1)) / headlineByYear.length;
  const y = (value) => margin.top + innerH - (value / max) * innerH;

  const bars = headlineByYear
    .map((d, i) => {
      const x = margin.left + i * (barW + barGap);
      const h = margin.top + innerH - y(d.headlineRows);
      const color = d.year >= 2022 ? palette.red : d.year >= 2020 ? palette.gold : palette.blue;
      const label = i % 2 === 0 || d.year >= 2022 ? `<text x="${x + barW / 2}" y="${height - 22}" text-anchor="middle" class="axis-label">${d.year}</text>` : "";
      return `
        <rect x="${x}" y="${y(d.headlineRows)}" width="${barW}" height="${h}" rx="3" fill="${color}"></rect>
        ${label}
      `;
    })
    .join("");

  const grid = [0, 1000, 2000, 3000, 4000]
    .map((tick) => {
      const ty = y(tick);
      return `
        <line x1="${margin.left}" x2="${width - margin.right}" y1="${ty}" y2="${ty}" class="grid-line"></line>
        <text x="${margin.left - 12}" y="${ty + 4}" text-anchor="end" class="axis-label">${tick}</text>
      `;
    })
    .join("");

  el.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" aria-hidden="true">
      ${grid}
      ${bars}
      <text x="${margin.left}" y="18" class="chart-label">Headline rows increase sharply in the final years</text>
      <text x="${width - margin.right}" y="18" text-anchor="end" class="chart-value">2023: 4,748 rows</text>
    </svg>
  `;
}

function renderRegimeCards() {
  const el = document.querySelector("#regimeCards");
  el.innerHTML = regimes
    .map(
      (d) => `
        <div class="regime-card">
          <strong>${d.name}</strong>
          <span>${d.days} days</span>
          <small>${d.avgHeadlines.toFixed(1)} headlines/day, ${pct(d.upDayRate, 1)} up-day rate, avg VIX ${d.avgVix.toFixed(1)}</small>
        </div>
      `
    )
    .join("");
}

function renderTercileChart() {
  const el = document.querySelector("#tercileChart");
  const width = 900;
  const height = 340;
  const margin = { top: 38, right: 28, bottom: 54, left: 58 };
  const panelW = (width - margin.left - margin.right - 70) / 2;
  const innerH = height - margin.top - margin.bottom;
  const maxReturn = 0.00062;
  const minRate = 0.5;
  const maxRate = 0.58;
  const colors = [palette.blue, palette.green, palette.red];
  const barW = 64;

  function barsFor(panelX, valueKey, scaleMax, scaleMin, formatter) {
    return terciles
      .map((d, i) => {
        const x = panelX + i * 98 + 42;
        const normalized = (d[valueKey] - scaleMin) / (scaleMax - scaleMin);
        const h = Math.max(0, normalized * innerH);
        const y = margin.top + innerH - h;
        return `
          <rect x="${x}" y="${y}" width="${barW}" height="${h}" rx="4" fill="${colors[i]}"></rect>
          <text x="${x + barW / 2}" y="${y - 8}" text-anchor="middle" class="chart-value">${formatter(d[valueKey])}</text>
          <text x="${x + barW / 2}" y="${height - 24}" text-anchor="middle" class="axis-label">${d.label}</text>
        `;
      })
      .join("");
  }

  const returnGrid = [0, 0.0003, 0.0006]
    .map((tick) => {
      const y = margin.top + innerH - (tick / maxReturn) * innerH;
      return `
        <line x1="${margin.left}" x2="${margin.left + panelW}" y1="${y}" y2="${y}" class="grid-line"></line>
        <text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" class="axis-label">${moneyPct(tick, 2)}</text>
      `;
    })
    .join("");

  const rateX = margin.left + panelW + 70;
  const rateGrid = [0.5, 0.54, 0.58]
    .map((tick) => {
      const y = margin.top + innerH - ((tick - minRate) / (maxRate - minRate)) * innerH;
      return `
        <line x1="${rateX}" x2="${rateX + panelW}" y1="${y}" y2="${y}" class="grid-line"></line>
        <text x="${rateX - 10}" y="${y + 4}" text-anchor="end" class="axis-label">${pct(tick, 0)}</text>
      `;
    })
    .join("");

  el.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" aria-hidden="true">
      <text x="${margin.left}" y="20" class="chart-label">Average next-day return</text>
      <text x="${rateX}" y="20" class="chart-label">Next-day up-rate</text>
      ${returnGrid}
      ${rateGrid}
      ${barsFor(margin.left, "avgReturn", maxReturn, 0, (v) => moneyPct(v, 3))}
      ${barsFor(rateX, "upDayRate", maxRate, minRate, (v) => pct(v, 1))}
    </svg>
  `;
}

function renderHorizontalBars(selector, data, metric, options = {}) {
  const el = document.querySelector(selector);
  const width = 920;
  const rowH = 48;
  const margin = { top: 34, right: 46, bottom: 30, left: 170 };
  const height = margin.top + margin.bottom + data.length * rowH;
  const values = data.map((d) => d[metric]);
  const min = options.min ?? Math.min(0, ...values);
  const max = options.max ?? Math.max(...values);
  const x = (value) => margin.left + ((value - min) / (max - min)) * (width - margin.left - margin.right);
  const zeroX = x(0);
  const baselineX = options.baseline !== undefined ? x(options.baseline) : null;
  const format = options.format ?? ((v) => v.toFixed(3));
  const color = options.color ?? palette.blue;

  const bars = data
    .map((d, i) => {
      const y = margin.top + i * rowH + 10;
      const value = d[metric];
      const x0 = Math.min(x(value), zeroX);
      const w = Math.abs(x(value) - zeroX);
      return `
        <text x="${margin.left - 14}" y="${y + 18}" text-anchor="end" class="chart-label">${d.label}</text>
        <rect x="${x0}" y="${y}" width="${Math.max(w, 2)}" height="24" rx="4" fill="${d.color || color}"></rect>
        <text x="${value >= 0 ? x(value) + 8 : x(value) - 8}" y="${y + 17}" text-anchor="${value >= 0 ? "start" : "end"}" class="chart-value">${format(value)}</text>
      `;
    })
    .join("");

  const baseline = baselineX
    ? `<line x1="${baselineX}" x2="${baselineX}" y1="${margin.top - 10}" y2="${height - margin.bottom + 6}" stroke="${palette.red}" stroke-width="2" stroke-dasharray="5 5"></line>
       <text x="${baselineX + 8}" y="${margin.top - 14}" class="chart-value">0.50 baseline</text>`
    : "";

  el.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" aria-hidden="true">
      <line x1="${zeroX}" x2="${zeroX}" y1="${margin.top - 8}" y2="${height - margin.bottom}" stroke="${palette.line}" stroke-width="1"></line>
      ${baseline}
      ${bars}
    </svg>
  `;
}

function renderModelChart(metric = "balanced_accuracy") {
  const labels = {
    accuracy: "Accuracy",
    balanced_accuracy: "Balanced accuracy",
    auc: "AUC"
  };
  renderHorizontalBars("#modelChart", modelMetrics, metric, {
    min: 0.46,
    max: 0.55,
    baseline: 0.5,
    format: (v) => `${(v * 100).toFixed(1)}%`,
    color: palette.blue
  });
  const panel = document.querySelector("#models .panel-header h3");
  panel.textContent = `Test-Set Model Comparison: ${labels[metric]}`;
}

function renderBacktestChart(metric = "strategy_cumulative_return") {
  const config = {
    strategy_cumulative_return: {
      min: 0,
      max: 0.145,
      format: (v) => pct(v, 1),
      color: palette.green,
      title: "Selected Test Backtests: Cumulative Return"
    },
    exposure: {
      min: 0,
      max: 1.05,
      format: (v) => pct(v, 1),
      color: palette.gold,
      title: "Selected Test Backtests: Market Exposure"
    },
    strategy_max_drawdown: {
      min: -0.3,
      max: 0,
      format: (v) => pct(v, 1),
      color: palette.red,
      title: "Selected Test Backtests: Maximum Drawdown"
    }
  }[metric];

  renderHorizontalBars("#backtestChart", backtests, metric, {
    min: config.min,
    max: config.max,
    format: config.format,
    color: config.color
  });
  document.querySelector("#backtest .panel-header h3").textContent = config.title;
}

function renderBacktestTable() {
  const rows = backtests
    .filter((row) => row.label !== "Buy & hold")
    .map(
      (row) => `
        <div class="mini-row">
          <span>${row.label}</span>
          <span>${pct(row.strategy_cumulative_return, 1)}</span>
          <span>${pct(row.exposure, 1)}</span>
          <span>${row.trades}</span>
        </div>
      `
    )
    .join("");

  document.querySelector("#backtestTable").innerHTML = `
    <div class="mini-row header">
      <span>Model</span>
      <span>Return</span>
      <span>Exposure</span>
      <span>Trades</span>
    </div>
    ${rows}
  `;
}

function renderRateHikeCards() {
  const el = document.querySelector("#rateHikeCards");
  el.innerHTML = rateHikeCards
    .map(
      (card) => `
        <div class="summary-card">
          <span>${card.label}</span>
          <strong>${card.value}</strong>
          <small>${card.note}</small>
        </div>
      `
    )
    .join("");
}

function renderSignalReality() {
  const el = document.querySelector("#signalReality");
  el.innerHTML = signalReality
    .map(
      (card) => `
        <div class="signal-card">
          <span>${card.label}</span>
          <strong>${card.value}</strong>
          <small>${card.note}</small>
        </div>
      `
    )
    .join("");
}

function initControls() {
  const modelButtons = Array.from(document.querySelectorAll("[data-model-metric]"));
  modelButtons.forEach((button) => {
    button.addEventListener("click", () => {
      setActiveButton(modelButtons, button);
      renderModelChart(button.dataset.modelMetric);
    });
  });

  const backtestButtons = Array.from(document.querySelectorAll("[data-backtest-metric]"));
  backtestButtons.forEach((button) => {
    button.addEventListener("click", () => {
      setActiveButton(backtestButtons, button);
      renderBacktestChart(button.dataset.backtestMetric);
    });
  });
}

function initNavigationPolish() {
  const progress = document.querySelector("#scrollProgress");
  const navLinks = Array.from(document.querySelectorAll(".nav a"));
  const sections = navLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  function updateProgress() {
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = scrollable > 0 ? window.scrollY / scrollable : 0;
    progress.style.width = `${Math.max(0, Math.min(1, ratio)) * 100}%`;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      navLinks.forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
      });
    },
    { rootMargin: "-20% 0px -60% 0px", threshold: [0.12, 0.2, 0.35] }
  );

  sections.forEach((section) => observer.observe(section));
  window.addEventListener("scroll", updateProgress, { passive: true });
  updateProgress();
}

renderHeadlineChart();
renderRegimeCards();
renderTercileChart();
renderModelChart();
renderBacktestChart();
renderBacktestTable();
renderRateHikeCards();
renderSignalReality();
initControls();
initNavigationPolish();
