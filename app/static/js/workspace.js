const root = document.querySelector(".workspace");
const scenarioId = root.dataset.scenarioId;

let map;
let populationLayer = L.layerGroup();
let facilityLayer = L.layerGroup();
let voronoiLayer = L.geoJSON(null);
let mapLegend = null;
let currentScenario = null;
let currentRun = null;
let chart = null;
let history = [];

const formatNumber = (value, digits = 2) => Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: digits });
const comparisonRows = [
  ["load_variance", "Load variance", (value) => formatNumber(value, 0)],
  ["max_load", "Max load", (value) => formatNumber(value, 0)],
  ["load_gini", "Load Gini", (value) => formatNumber(value, 3)],
  ["outside_radius_percent", "Outside-radius %", (value) => `${formatNumber(100 * value, 1)}%`],
  ["mean_weighted_distance", "Mean distance", (value) => formatNumber(value, 4)]
];
const metricExplanations = [
  ["Population", "The total number of people represented by the uploaded demand points."],
  ["Facilities", "The number of clinics, hospitals, or service locations being evaluated."],
  ["Mean load", "The average population assigned to each facility if demand were evenly distributed."],
  ["Load variance", "How unevenly population demand is distributed across facilities. Lower values suggest a more balanced system."],
  ["Outside-radius percentage", "The share of population farther than the acceptable radius from its assigned facility. Lower values suggest better geographic access."],
  ["Mean distance", "The average distance from population points to their assigned facilities, weighted by population size."],
  ["Load Gini", "A fairness index for facility load. Zero is perfectly even; higher values indicate more inequality across facilities."],
  ["Max load", "The largest population assigned to any one facility. This highlights potential pressure points in the network."]
];

function parametersFromForm() {
  return {
    max_iterations: Number(document.querySelector("#max_iterations").value),
    acceptable_radius: Number(document.querySelector("#acceptable_radius").value),
    fuzzy_radius: Number(document.querySelector("#fuzzy_radius").value),
    omega: Number(document.querySelector("#omega").value),
    learning_rate: Number(document.querySelector("#learning_rate").value),
    tolerance: 0.00001,
    outside_radius_weight: 1,
    respect_fixed_facilities: true,
    empty_cluster_strategy: "keep"
  };
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed.");
  return data;
}

async function loadScenario() {
  currentScenario = await fetchJson(`/api/scenarios/${scenarioId}`);
  document.querySelector("#scenario-name").textContent = currentScenario.name;
  document.querySelector("#acceptable_radius").value = currentScenario.parameters.acceptable_radius ?? 0.05;
  document.querySelector("#fuzzy_radius").value = currentScenario.parameters.fuzzy_radius ?? 0.05;
  document.querySelector("#omega").value = currentScenario.parameters.omega ?? 0.2;
  initializeMap();
  renderPopulation(currentScenario.population);
  renderFacilities(currentScenario.facilities, currentScenario.baseline.facility_scores, currentScenario.baseline.metrics);
  renderVoronoi(currentScenario.baseline.voronoi);
  renderMetrics(currentScenario.baseline.metrics);
  renderScores(currentScenario.baseline.facility_scores);
  renderComparison(currentScenario.baseline.metrics, currentScenario.baseline.metrics, false);
  renderExplainResults(currentScenario.baseline.metrics, currentScenario.baseline.facility_scores, false);
  await createRun();
}

function initializeMap() {
  if (map) return;
  const b = currentScenario.bounds;
  map = L.map("map").fitBounds([[b.min_lat, b.min_lon], [b.max_lat, b.max_lon]]);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(map);
  voronoiLayer.addTo(map);
  populationLayer.addTo(map);
  facilityLayer.addTo(map);
  addMapLegend();
}

function renderPopulation(points) {
  populationLayer.clearLayers();
  const maxPopulation = Math.max(...points.map((p) => p.population), 1);
  points.forEach((point) => {
    const radius = 4 + 16 * Math.sqrt(point.population / maxPopulation);
    L.circleMarker([point.lat, point.lon], {
      radius,
      color: "#275d8c",
      fillColor: "#3f8ecb",
      fillOpacity: 0.36,
      weight: 1
    }).bindTooltip(`${point.name}: ${formatNumber(point.population, 0)}`).addTo(populationLayer);
  });
}

function renderFacilities(facilities, scores = [], metrics = {}) {
  facilityLayer.clearLayers();
  const scoreById = Object.fromEntries(scores.map((row) => [row.facility_id, row]));
  const meanLoad = Number(metrics.mean_load || 0);
  const maxLoad = Math.max(...scores.map((row) => Number(row.regular_population || 0)), meanLoad, 1);
  facilities.forEach((facility) => {
    const score = scoreById[facility.id] || {
      name: facility.name,
      regular_population: 0,
      outside_radius_population: 0,
      score: 0
    };
    const marker = L.marker([facility.lat, facility.lon], {
      draggable: true,
      icon: facilityIcon(score, meanLoad, maxLoad)
    }).bindTooltip(facilityTooltip(score, meanLoad), {
      className: "facility-detail-tooltip",
      direction: "top",
      opacity: 0.96
    });
    marker.on("dragend", async () => {
      const position = marker.getLatLng();
      facility.lat = position.lat;
      facility.lon = position.lng;
      await saveFacilityPositions();
    });
    marker.addTo(facilityLayer);
  });
}

function facilityIcon(score, meanLoad, maxLoad) {
  const assigned = Number(score.regular_population || 0);
  const outside = Number(score.outside_radius_population || 0);
  const overloaded = meanLoad > 0 && assigned >= meanLoad * 1.5;
  const loadRatio = Math.max(0, assigned / maxLoad);
  const size = Math.round(18 + 22 * Math.sqrt(loadRatio));
  const background = overloaded ? "#8f1d16" : assigned > meanLoad ? "#b8462e" : "#1b6a5c";
  const border = outside > 0 ? "#f2b84b" : "#ffffff";
  const shadow = overloaded ? "0 0 0 5px rgba(214, 75, 58, 0.26)" : "0 2px 8px rgba(23, 32, 42, 0.28)";

  return L.divIcon({
    className: "facility-map-icon",
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    html: `
      <span
        class="facility-symbol ${overloaded ? "facility-symbol-overloaded" : ""}"
        style="width:${size}px;height:${size}px;background:${background};border-color:${border};box-shadow:${shadow};"
        aria-label="Facility marker"
      ></span>
    `
  });
}

function facilityTooltip(score, meanLoad) {
  const assigned = Number(score.regular_population || 0);
  const outside = Number(score.outside_radius_population || 0);
  const overloaded = meanLoad > 0 && assigned >= meanLoad * 1.5;
  return `
    <strong>${escapeHtml(score.name)}</strong>
    <dl class="map-tooltip-metrics">
      <div><dt>Assigned population</dt><dd>${formatNumber(assigned, 0)}</dd></div>
      <div><dt>Outside-radius demand</dt><dd>${formatNumber(outside, 0)}</dd></div>
      <div><dt>Score</dt><dd>${formatNumber(score.score, 0)}</dd></div>
      <div><dt>Status</dt><dd>${overloaded ? "Overloaded: above 150% of mean load" : "Within high-load threshold"}</dd></div>
    </dl>
  `;
}

function renderVoronoi(geojson) {
  voronoiLayer.clearLayers();
  voronoiLayer = L.geoJSON(geojson, {
    style: (feature) => ({
      color: "#33434f",
      weight: 1,
      fillColor: feature.properties.color || "#dfe7ed",
      fillOpacity: 0.42
    }),
    onEachFeature: (feature, layer) => {
      layer.bindTooltip(`${feature.properties.name}<br>Score: ${formatNumber(feature.properties.score || 0, 0)}`);
    }
  }).addTo(map);
}

function addMapLegend() {
  if (mapLegend) return;
  mapLegend = L.control({ position: "bottomright" });
  mapLegend.onAdd = () => {
    const div = L.DomUtil.create("div", "map-legend");
    div.innerHTML = `
      <h4>Map Legend</h4>
      <div><span class="legend-symbol legend-facility"></span> Facility marker</div>
      <div><span class="legend-symbol legend-population"></span> Population point</div>
      <div><span class="legend-symbol legend-region"></span> Voronoi service region</div>
      <div><span class="legend-symbol legend-overloaded"></span> Overloaded facility (&gt;150% mean load)</div>
      <div><span class="legend-symbol legend-outside"></span> Outside-radius demand present</div>
    `;
    L.DomEvent.disableClickPropagation(div);
    return div;
  };
  mapLegend.addTo(map);
}

async function saveFacilityPositions() {
  const data = await fetchJson(`/api/scenarios/${scenarioId}/facilities`, {
    method: "PUT",
    body: JSON.stringify({ facilities: currentScenario.facilities })
  });
  currentScenario = data;
  renderVoronoi(data.baseline.voronoi);
  renderFacilities(data.facilities, data.baseline.facility_scores, data.baseline.metrics);
  renderMetrics(data.baseline.metrics);
  renderScores(data.baseline.facility_scores);
  renderComparison(data.baseline.metrics, data.baseline.metrics, false);
  renderExplainResults(data.baseline.metrics, data.baseline.facility_scores, false);
  await createRun();
}

function renderMetrics(metrics) {
  const rows = [
    ["Population", formatNumber(metrics.total_population, 0)],
    ["Facilities", formatNumber(metrics.facility_count, 0)],
    ["Mean load", formatNumber(metrics.mean_load, 0)],
    ["Load variance", formatNumber(metrics.load_variance, 0)],
    ["Outside radius", `${formatNumber(100 * metrics.outside_radius_percent, 1)}%`],
    ["Mean distance", formatNumber(metrics.mean_weighted_distance, 4)],
    ["Load Gini", formatNumber(metrics.load_gini, 3)],
    ["Max load", formatNumber(metrics.max_load, 0)]
  ];
  document.querySelector("#metrics").innerHTML = rows.map(([label, value]) => `
    <div class="metric"><strong>${label}</strong><span>${value}</span></div>
  `).join("");
}

function renderScores(scores) {
  document.querySelector("#facility-table").innerHTML = scores.map((row) => `
    <tr>
      <td>${row.name}</td>
      <td>${formatNumber(row.regular_population, 0)}</td>
      <td>${formatNumber(row.outside_radius_population, 0)}</td>
      <td>${formatNumber(row.score, 0)}</td>
    </tr>
  `).join("");
}

function renderComparison(baselineMetrics, optimizedMetrics, hasOptimized) {
  document.querySelector("#baseline-comparison").innerHTML = comparisonMarkup(baselineMetrics);
  document.querySelector("#optimized-comparison").innerHTML = comparisonMarkup(optimizedMetrics);

  const summary = document.querySelector("#comparison-summary");
  if (!hasOptimized) {
    summary.textContent = "Run the optimizer to compare baseline and optimized placement.";
    return;
  }

  const varianceReduction = percentReduction(baselineMetrics.load_variance, optimizedMetrics.load_variance);
  const outsideReduction = percentReduction(
    baselineMetrics.outside_radius_population,
    optimizedMetrics.outside_radius_population
  );
  summary.textContent = `Optimization reduced load variance by ${formatSignedPercent(varianceReduction)} and outside-radius population by ${formatSignedPercent(outsideReduction)}.`;
}

function renderExplainResults(metrics, scores, hasOptimized) {
  document.querySelector("#metric-explanations").innerHTML = metricExplanations.map(([term, definition]) => `
    <div>
      <dt>${term}</dt>
      <dd>${definition}</dd>
    </div>
  `).join("");
  document.querySelector("#scenario-interpretation").innerHTML = interpretationMarkup(metrics, scores, hasOptimized);
}

function interpretationMarkup(metrics, scores, hasOptimized) {
  const meanLoad = Number(metrics.mean_load || 0);
  const overloaded = scores
    .filter((row) => Number(row.regular_population || 0) > meanLoad * 1.15)
    .sort((a, b) => b.regular_population - a.regular_population)
    .slice(0, 3);
  const underserved = scores
    .filter((row) => Number(row.outside_radius_population || 0) > 0)
    .sort((a, b) => b.outside_radius_population - a.outside_radius_population)
    .slice(0, 3);

  const loadText = overloaded.length
    ? `${joinNames(overloaded)} appear to carry above-average demand and may need added capacity, staffing, or nearby support.`
    : "No facility is currently far above the average assigned load, suggesting demand is relatively balanced by this measure.";
  const accessText = underserved.length
    ? `${joinNames(underserved)} have the largest assigned populations outside the acceptable radius, indicating areas where geographic access may be weaker.`
    : "The model does not identify a large outside-radius burden, suggesting most represented population points are within the selected access threshold.";
  const equityText = equityInterpretation(metrics, hasOptimized);

  return `
    <ul class="interpretation-list">
      <li>${loadText}</li>
      <li>${accessText}</li>
      <li>${equityText}</li>
    </ul>
  `;
}

function joinNames(rows) {
  return rows.map((row) => row.name).join(", ");
}

function equityInterpretation(metrics, hasOptimized) {
  const gini = Number(metrics.load_gini || 0);
  const outsidePercent = Number(metrics.outside_radius_percent || 0);
  const stage = hasOptimized ? "After optimization" : "Before optimization";
  const equity = gini < 0.15 ? "fairly even" : gini < 0.3 ? "moderately uneven" : "highly uneven";
  const access = outsidePercent < 0.1 ? "strong access by the selected radius" : outsidePercent < 0.25 ? "some access gaps" : "substantial access gaps";
  return `${stage}, facility load appears ${equity}, and the outside-radius percentage suggests ${access}. Use this as a planning signal rather than a final public-health decision.`;
}

function comparisonMarkup(metrics) {
  return comparisonRows.map(([key, label, formatter]) => `
    <div>
      <dt>${label}</dt>
      <dd>${formatter(metrics[key])}</dd>
    </div>
  `).join("");
}

function percentReduction(before, after) {
  const start = Number(before || 0);
  const end = Number(after || 0);
  if (start === 0) return end === 0 ? 0 : -100;
  return ((start - end) / start) * 100;
}

function formatSignedPercent(value) {
  return `${formatNumber(value, 1)}%`;
}

async function createRun() {
  const data = await fetchJson(`/api/scenarios/${scenarioId}/runs`, {
    method: "POST",
    body: JSON.stringify({ algorithm: "weighted_lloyd", parameters: parametersFromForm() })
  });
  currentRun = data;
  history = [];
  applyRun(data);
  updateExportLinks();
}

function applyRun(run) {
  currentRun = run;
  const latest = run.latest;
  const facilities = latest.facilities.ids.map((id, index) => ({
    id,
    name: latest.facilities.names[index],
    lat: latest.facilities.lat[index],
    lon: latest.facilities.lon[index],
    risk_factor: latest.facilities.risk_factor[index],
    fixed: latest.facilities.fixed[index]
  }));
  currentScenario.facilities = facilities;
  renderFacilities(facilities, latest.facility_scores, latest.metrics);
  renderVoronoi(run.voronoi);
  renderMetrics(latest.metrics);
  renderScores(latest.facility_scores);
  renderComparison(currentScenario.baseline.metrics, latest.metrics, latest.iteration > 0);
  renderExplainResults(latest.metrics, latest.facility_scores, latest.iteration > 0);
  history.push({
    iteration: latest.iteration,
    variance: latest.metrics.load_variance,
    distance: latest.metrics.mean_weighted_distance
  });
  renderChart();
}

function renderChart() {
  const context = document.querySelector("#history-chart");
  const labels = history.map((row) => row.iteration);
  const variance = history.map((row) => row.variance);
  if (!chart) {
    chart = new Chart(context, {
      type: "line",
      data: {
        labels,
        datasets: [{ label: "Load variance", data: variance, borderColor: "#1b6a5c", tension: 0.25 }]
      },
      options: { responsive: true, plugins: { legend: { display: false } } }
    });
    return;
  }
  chart.data.labels = labels;
  chart.data.datasets[0].data = variance;
  chart.update();
}

function updateExportLinks() {
  if (!currentRun) return;
  document.querySelector("#export-facilities").href = `/api/runs/${currentRun.id}/export/facilities.csv`;
  document.querySelector("#export-metrics").href = `/api/runs/${currentRun.id}/export/metrics.csv`;
  document.querySelector("#export-report-csv").href = `/api/runs/${currentRun.id}/export/report.csv`;
  document.querySelector("#export-report-pdf").href = `/api/runs/${currentRun.id}/export/report.pdf`;
  document.querySelector("#export-json").href = `/api/runs/${currentRun.id}/export/scenario.json`;
}

async function stepRun() {
  if (!currentRun) await createRun();
  const data = await fetchJson(`/api/runs/${currentRun.id}/step`, { method: "POST", body: "{}" });
  applyRun(data);
}

async function convergeRun() {
  if (!currentRun) await createRun();
  const data = await fetchJson(`/api/runs/${currentRun.id}/run-to-convergence`, {
    method: "POST",
    body: JSON.stringify({ max_steps: Number(document.querySelector("#max_iterations").value) })
  });
  applyRun(data);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

document.querySelector("#create-run").addEventListener("click", createRun);
document.querySelector("#step-run").addEventListener("click", stepRun);
document.querySelector("#converge-run").addEventListener("click", convergeRun);

loadScenario().catch((error) => {
  document.body.innerHTML = `<main class="home"><h1>Could not load workspace</h1><p>${error.message}</p></main>`;
});
