import { guidanceValues, planBatch } from "./batch.js";

const $ = (id) => document.getElementById(id);
const names = { hunyuan_shape: "Hunyuan3D 2.1", triposr: "TripoSR", hunyuan_multiview: "Hunyuan Multi-view", ultrashape: "UltraShape" };
const generators = ["hunyuan_shape", "triposr"];
let refinementReady = false, multiviewReady = false;
const isMultiview = () => $("input-mode").value === "multiview";
const hunyuanBackend = () => isMultiview() ? "hunyuan_multiview" : "hunyuan_shape";
const settingsPanel = name => $(`params-${name === "hunyuan_multiview" ? "hunyuan_shape" : name}`);

// Refinement is submitted from finished meshes; these settings stay separate from generation.
const refinement = document.createElement("details");
refinement.id = "refinement-settings";
refinement.className = "backend";
refinement.innerHTML = `<summary>UltraShape refinement settings</summary>
  <p class="hint">Use “Refine with UltraShape” on a finished mesh. A new job uses its mesh and reference image; the original stays available.</p>
  <p class="hint" id="refinement-status">Checking installation…</p>
  <fieldset id="params-ultrashape"><div class="param-grid"></div>
  <label><input name="cpu_offload" type="checkbox" checked> CPU offload (recommended)</label>
  <label><input name="remove_background" type="checkbox" checked> Remove background</label>
  <label><input name="experimental_resolution" type="checkbox"> Experimental resolution</label>
  <p class="hint">Start with 8192 geometry tokens and 256 resolution. More tokens preserve more structure and use much more memory. This TITAN profile uses FP16, PyTorch attention and CPU mesh extraction.</p></fieldset>`;
document.querySelector('[data-backend="triposr"]').after(refinement);
for (const [name, label, value, min, max, step] of [
  ["steps", "Sampling steps", 30, 1, 100, 1], ["guidance_scale", "Guidance", 5, 0, 30, 0.1],
  ["seed", "Seed", 1234, 0, 4294967295, 1], ["octree_resolution", "Mesh resolution", 256, 16, 1024, 1],
  ["num_chunks", "Decoder chunk size", 2048, 256, 65536, 1], ["num_latents", "Geometry tokens", 8192, 1024, 32768, 1024],
  ["normalize_scale", "Normalisation scale", 0.99, 0.5, 0.9999, 0.001],
]) {
  const wrapper = document.createElement("label");
  wrapper.textContent = label;
  const input = document.createElement("input");
  Object.assign(input, {name, type: "number", value, min, max, step, required: true});
  wrapper.append(input);
  refinement.querySelector(".param-grid").append(wrapper);
}
const terminal = new Set(["succeeded", "failed", "cancelled"]);
let inputs = [],
  groups = [],
  jobs = [],
  selectedJob = null,
  viewer,
  createViewer,
  ready = false,
  submitting = false,
  refreshing = false;
let uploadLimit = 32 * 1048576,
  groupLimit = 20,
  groupTotal = 0,
  queueOnly = false,
  thumbViewer,
  thumbBusy = false;
let lastRenderSignature = "";
const cards = new Map(),
  thumbQueue = [],
  thumbPending = new Set(),
  thumbURLs = new Map(),
  collapsed = new Set();
const formControls = [
  ...$("generate-form").querySelectorAll("input:not([type=file]),select"),
].filter((input) => input.id !== "batch-name" && input.id !== "preset");
const keyOf = (input) =>
  input.name ? `${input.closest("fieldset").id}:${input.name}` : input.id;
const getSettings = () =>
  Object.fromEntries(
    formControls.map((input) => [
      keyOf(input),
      input.type === "checkbox" ? input.checked : input.value,
    ]),
  );
const defaults = getSettings();
function stored(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}
function save(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    notice(
      "Browser storage is unavailable. Settings will last for this session.",
    );
  }
}
function notice(text, error = false) {
  $("notice").hidden = !text;
  $("notice").textContent = text;
  $("notice").setAttribute("role", error ? "alert" : "status");
}
function node(tag, cls, text) {
  const item = document.createElement(tag);
  if (cls) item.className = cls;
  if (text !== undefined) item.textContent = text;
  return item;
}
function button(text, action, cls = "") {
  const item = node("button", cls, text);
  item.type = "button";
  item.addEventListener("click", action);
  return item;
}
async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const data = await response.json();
      message =
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail || data);
    } catch {}
    throw new Error(message);
  }
  return response.json();
}
const previewURL = (job) => `/v1/jobs/${job.job_id}/preview`;
const artifactURL = (job) => `/v1/jobs/${job.job_id}/artifact`;
const sourceURL = (job) => `/v1/assets/${job.asset_id}/image`;
const randomSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];
const selectedBackends = () => isMultiview() ? (multiviewReady ? ["hunyuan_multiview"] : []) :
  generators.filter(
    (name) => $(`select-${name}`).checked && !$(`select-${name}`).disabled,
  );
function parameters(name) {
  return Object.fromEntries(
    [...settingsPanel(name).querySelectorAll("[name]")].map((input) => [
      input.name,
      input.type === "checkbox" ? input.checked : Number(input.value),
    ]),
  );
}
function applySettings(settings) {
  for (const input of formControls) {
    const value = settings[keyOf(input)];
    if (value === undefined) continue;
    if (input.type === "checkbox") input.checked = Boolean(value);
    else input.value = String(value);
  }
  updateForm();
}
const presetValues = {
  draft: { steps: 15, octree_resolution: 128, mc_resolution: 128 },
  balanced: { steps: 30, octree_resolution: 256, mc_resolution: 256 },
  detail: { steps: 40, octree_resolution: 384, mc_resolution: 384 },
};
function populatePresets() {
  const saved = stored("meshtitan-presets", {});
  for (const option of [...$("preset").options])
    if (option.value.startsWith("saved:")) option.remove();
  for (const name of Object.keys(saved))
    $("preset").append(new Option(name, `saved:${name}`));
}
populatePresets();
applySettings(stored("meshtitan-settings", defaults));
if (stored("meshtitan-settings", null)) $("preset").value = "custom";
$("preset").addEventListener("change", () => {
  const value = $("preset").value;
  if (value.startsWith("saved:"))
    applySettings(stored("meshtitan-presets", {})[value.slice(6)] || {});
  else if (presetValues[value]) {
    const preset = presetValues[value];
    for (const [name, val] of Object.entries(preset))
      document.querySelector(`[name="${name}"]`).value = val;
  }
  updateForm();
  save("meshtitan-settings", getSettings());
});
$("save-settings").addEventListener("click", () => {
  const name = prompt("Name this preset", "My preset")?.trim();
  if (!name) return;
  const presets = stored("meshtitan-presets", {});
  Object.defineProperty(presets, name, {
    value: getSettings(),
    enumerable: true,
    configurable: true,
    writable: true,
  });
  save("meshtitan-presets", presets);
  populatePresets();
  $("preset").value = `saved:${name}`;
  notice(`Saved preset “${name}”.`);
});
$("reset-settings").addEventListener("click", () => {
  applySettings(defaults);
  $("preset").value = "balanced";
  save("meshtitan-settings", getSettings());
});
for (const input of formControls) {
  input.addEventListener("input", updateForm);
  input.addEventListener("change", () => {
    if (input.name) $("preset").value = "custom";
    updateForm();
    save("meshtitan-settings", getSettings());
  });
}
$("random-seed").addEventListener("click", () => {
  document.querySelector("[name=seed]").value = randomSeed();
  save("meshtitan-settings", getSettings());
});
$("generate-form").addEventListener(
  "invalid",
  (event) => {
    let detail = event.target.closest("details");
    while (detail) {
      detail.open = true;
      detail = detail.parentElement.closest("details");
    }
  },
  true,
);
function plannedCount() {
  const backends = selectedBackends();
  const sweepCount = $("guidance-sweep").checked
    ? Number($("guidance-count").value) || 1
    : 1;
  return (
    (isMultiview() ? (inputs.length ? 1 : 0) : inputs.length) *
    ((backends.some(name => name.startsWith("hunyuan_"))
      ? (Number($("variations").value) || 1) * sweepCount
      : 0) +
      (backends.includes("triposr") ? 1 : 0))
  );
}
function updateForm() {
  for (const name of generators)
    $(`params-${name}`).disabled =
      !$(`select-${name}`).checked || $(`select-${name}`).disabled;
  const hunyuanEnabled = selectedBackends().includes(hunyuanBackend());
  $("select-hunyuan_shape").hidden = isMultiview();
  $("view-hint").hidden = !isMultiview();
  document.querySelector('[data-backend="triposr"]').hidden = isMultiview();
  document.querySelector('[data-backend="hunyuan_shape"] strong').textContent = isMultiview() ? "Hunyuan Multi-view · 2mv" : "Hunyuan3D 2.1";
  if (isMultiview()) $("params-hunyuan_shape").disabled = !multiviewReady;
  $("guidance-count").disabled = $("guidance-step").disabled =
    !hunyuanEnabled || !$("guidance-sweep").checked;
  let issue = "";
  try {
    const values = guidanceValues(
      parameters(hunyuanBackend()).guidance_scale,
      {
        enabled: hunyuanEnabled && $("guidance-sweep").checked,
        count: Number($("guidance-count").value),
        step: Number($("guidance-step").value),
      },
    );
    $("guidance-summary").textContent =
      hunyuanEnabled && $("guidance-sweep").checked
        ? `Guidance: ${values.join(", ")} · same seed at every strength.`
        : "Each seed is held fixed across all guidance values.";
    for (const backend of selectedBackends()) {
      const params = parameters(backend);
      const resolution = params.octree_resolution ?? params.mc_resolution;
      if (
        (resolution < 64 || resolution > 512) &&
        !params.experimental_resolution
      )
        issue = `${names[backend]}: enable experimental resolutions for ${resolution}.`;
    }
  } catch (error) {
    issue = error.message;
    $("guidance-summary").textContent = issue;
  }
  if (isMultiview()) {
    const views = inputs.map(item => item.view);
    if (inputs.length < 2 || inputs.length > 4 || !views.includes("front") || new Set(views).size !== views.length)
      issue = "Multi-view needs 2–4 images with unique directions, including front.";
    if (!multiviewReady) issue = "Multi-view is unavailable. Run windows.bat setup and restart the server.";
  }
  const count = plannedCount();
  $("generate").disabled =
    !ready ||
    !inputs.length ||
    !selectedBackends().length ||
    submitting ||
    count > 200 ||
    Boolean(issue);
  $("generate").textContent = submitting
    ? "Preparing batch…"
    : `Generate ${count || ""} ${count === 1 ? "mesh" : "meshes"} ↗`;
  $("queue-estimate").textContent =
    issue ||
    (count > 200
      ? "Maximum 200 jobs per batch. Reduce images or variations."
      : inputs.length
        ? `${inputs.length} ${isMultiview() ? "views" : inputs.length === 1 ? "image" : "images"} · ${count} ${count === 1 ? "job" : "jobs"} · sequential processing`
        : "Add images to start a batch.");
  $("input-count").textContent = inputs.length;
}
async function addFiles(files) {
  const accepted = [],
    errors = [];
  for (const file of files) {
    if (inputs.length + accepted.length >= 100) {
      errors.push("Maximum 100 input images.");
      break;
    }
    if (!/\.(png|jpe?g|webp)$/i.test(file.name)) {
      errors.push(`${file.name}: use PNG, JPG or WebP.`);
      continue;
    }
    if (file.size > uploadLimit) {
      errors.push(`${file.name}: exceeds ${uploadLimit / 1048576} MB.`);
      continue;
    }
    const url = URL.createObjectURL(file);
    try {
      await new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = resolve;
        image.onerror = reject;
        image.src = url;
      });
      accepted.push({ id: crypto.randomUUID(), file, url, view: ["front", "left", "back", "right"][(inputs.length + accepted.length) % 4] });
    } catch {
      URL.revokeObjectURL(url);
      errors.push(`${file.name}: could not open image.`);
    }
  }
  inputs.push(...accepted);
  renderInputs();
  notice(
    errors.length
      ? errors.join(" ")
      : `${accepted.length} image${accepted.length === 1 ? "" : "s"} added.`,
    Boolean(errors.length),
  );
  $("image-file").value = "";
}
function renderInputs() {
  $("input-images").replaceChildren();
  for (const input of inputs) {
    const tile = node("div", "input-tile");
    const image = node("img");
    image.src = input.url;
    image.alt = input.file.name;
    tile.append(image, node("span", "", input.file.name));
    if (isMultiview()) {
      const view = node("select");
      view.setAttribute("aria-label", `View direction for ${input.file.name}`);
      for (const direction of ["front", "left", "back", "right"])
        view.append(new Option(direction, direction));
      view.value = input.view;
      view.addEventListener("change", () => { input.view = view.value; updateForm(); });
      tile.append(view);
    }
    const remove = button("×", () => {
      inputs = inputs.filter((item) => item.id !== input.id);
      URL.revokeObjectURL(input.url);
      renderInputs();
    });
    remove.setAttribute("aria-label", `Remove ${input.file.name}`);
    tile.append(remove);
    $("input-images").append(tile);
  }
  updateForm();
}
$("input-mode").addEventListener("change", () => {
  if (isMultiview()) $("params-hunyuan_shape").querySelector('[name="cpu_offload"]').checked = true;
  renderInputs();
  save("meshtitan-settings", getSettings());
});
$("image-file").addEventListener("change", (event) =>
  addFiles([...event.target.files]),
);
for (const type of ["dragenter", "dragover"])
  $("dropzone").addEventListener(type, (event) => {
    event.preventDefault();
    $("dropzone").classList.add("dragover");
  });
for (const type of ["dragleave", "drop"])
  $("dropzone").addEventListener(type, (event) => {
    event.preventDefault();
    $("dropzone").classList.remove("dragover");
    if (type === "drop") addFiles([...event.dataTransfer.files]);
  });
$("clear-inputs").addEventListener("click", () => {
  inputs.forEach((input) => URL.revokeObjectURL(input.url));
  inputs = [];
  renderInputs();
});
$("generate-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submitting || !ready || !inputs.length || !selectedBackends().length)
    return;
  const selected = selectedBackends(),
    images = [...inputs],
    format = $("output-format").value,
    repeats = Number($("variations").value),
    mode = $("seed-mode").value;
  const sweep = {
    enabled: $("guidance-sweep").checked,
    count: Number($("guidance-count").value),
    step: Number($("guidance-step").value),
  };
  const settings = Object.fromEntries(
    selected.map((name) => [name, parameters(name)]),
  );
  const name =
    $("batch-name").value.trim() ||
    `Batch · ${new Date().toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`;
  if (plannedCount() > 200) return;
  submitting = true;
  updateForm();
  try {
    const assets = [];
    // Validate the sweep before uploading files or queuing any work.
    for (const name of selected.filter(name => name.startsWith("hunyuan_")))
      guidanceValues(settings[name].guidance_scale, sweep);
    for (let i = 0; i < images.length; i++) {
      notice(`Uploading ${i + 1} / ${images.length}: ${images[i].file.name}`);
      const data = new FormData();
      data.append("file", images[i].file);
      const asset = await api("/v1/assets", { method: "POST", body: data });
      assets.push({
        asset_id: asset.asset_id,
        source_name: images[i].file.name,
        output_format: format,
        view: images[i].view,
      });
    }
    const requests = planBatch({
      assets,
      backends: selected,
      settings,
      variations: repeats,
      seedMode: mode,
      sweep,
      randomSeed,
      multiview: isMultiview(),
    });
    const records = await api("/v1/batches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, jobs: requests }),
    });
    notice(
      `“${name}” queued: ${records.length} jobs. Images will be processed in import order.`,
    );
    groupLimit = Math.max(groupLimit, 20);
    await refreshGallery();
  } catch (error) {
    notice(`Batch was not submitted: ${error.message}`, true);
  } finally {
    submitting = false;
    updateForm();
  }
});
function statusCounts(records) {
  return records.reduce((result, job) => {
    result[job.status] = (result[job.status] || 0) + 1;
    return result;
  }, {});
}
function selectedFilter(job, group) {
  const search = $("search").value.toLowerCase().trim(),
    status = $("status-filter").value;
  return (
    (!queueOnly || !terminal.has(job.status)) &&
    (status === "all" || job.status === status) &&
    (!search ||
      `${group.name} ${job.source_name} ${names[job.backend]} ${job.job_id} ${job.parameters.seed ?? ""}`
        .toLowerCase()
        .includes(search))
  );
}
function downloadLink(job) {
  const link = node("a", "download", `${job.output_format.toUpperCase()} ↓`);
  link.href = artifactURL(job);
  link.download = `${job.backend}-${job.job_id.slice(0, 8)}.${job.output_format}`;
  return link;
}
function reuseSettings(job) {
  if (job.backend === "ultrashape") $("refinement-settings").open = true;
  else $("input-mode").value = job.backend === "hunyuan_multiview" ? "multiview" : "batch";
  renderInputs();
  const effective = job.result?.backend?.parameters || job.parameters;
  for (const name of generators)
    $(`select-${name}`).checked = name === job.backend || (name === "hunyuan_shape" && job.backend === "hunyuan_multiview");
  for (const input of settingsPanel(job.backend).querySelectorAll("[name]"))
    if (effective[input.name] !== undefined) {
      if (input.type === "checkbox") input.checked = effective[input.name];
      else input.value = effective[input.name];
    }
  $("output-format").value = job.output_format;
  $("preset").value = "custom";
  updateForm();
  save("meshtitan-settings", getSettings());
  notice(
    "Settings copied to the controls. Choose images and generate when ready.",
  );
}
async function reuseSource(job) {
  try {
    const views = Object.entries(job.view_assets || {});
    if (views.length) {
      $("input-mode").value = "multiview";
      const files = [];
      for (const [direction, assetId] of views) {
        const response = await fetch(`/v1/assets/${assetId}/image`);
        if (!response.ok) throw new Error(`Source ${direction} image is unavailable.`);
        const blob = await response.blob();
        const extension = blob.type === "image/jpeg" ? "jpg" : blob.type === "image/webp" ? "webp" : "png";
        files.push(new File([blob], `${direction}.${extension}`, {type: blob.type}));
      }
      const previousLength = inputs.length;
      await addFiles(files);
      inputs.slice(previousLength).forEach((input, i) => { input.view = views[i][0]; });
      renderInputs();
    } else {
      const response = await fetch(sourceURL(job));
      if (!response.ok) throw new Error("Source image is unavailable.");
      const blob = await response.blob();
      await addFiles([
        new File([blob], job.source_name || "input.png", { type: blob.type }),
      ]);
    }
  } catch (error) {
    notice(error.message, true);
  }
}
async function cancelJob(job) {
  try {
    await api(`/v1/jobs/${job.job_id}/cancel`, { method: "POST" });
    await refreshGallery();
  } catch (error) {
    notice(error.message, true);
  }
}
async function refineJob(job) {
  try {
    $("refinement-settings").open = true;
    const invalid = [...$("params-ultrashape").querySelectorAll("input")].find(input => !input.checkValidity());
    if (invalid) { invalid.reportValidity(); return; }
    const params = parameters("ultrashape");
    if ((params.octree_resolution < 64 || params.octree_resolution > 512) && !params.experimental_resolution)
      throw new Error("Enable experimental resolutions in UltraShape settings for this resolution.");
    const records = await api("/v1/batches", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name: `Refine · ${job.source_name || job.job_id.slice(0,8)}`, jobs: [{
        backend: "ultrashape", asset_id: job.asset_id, parent_job_id: job.job_id,
        source_name: job.source_name, output_format: $("output-format").value, parameters: params,
      }]}),
    });
    notice(`UltraShape refinement queued: ${records[0].job_id.slice(0,8)}. Source mesh preserved.`);
    await refreshGallery();
  } catch (error) { notice(error.message, true); }
}
async function cancelBatch(group) {
  try {
    await api(`/v1/batches/${group.id}/cancel`, { method: "POST" });
    await refreshGallery();
  } catch (error) {
    notice(error.message, true);
  }
}
function applyThumbnail(job, url) {
  const card = cards.get(job.job_id)?.node;
  if (!card) return;
  const image = card.querySelector(".thumbnail-button > img");
  if (image) image.src = url;
  const label = card.querySelector(".thumb-label");
  if (label) {
    label.textContent = "MESH";
    label.classList.add("mesh");
  }
  let corner = card.querySelector(".source-corner");
  if (!corner) {
    corner = node("img", "source-corner");
    corner.src = sourceURL(job);
    corner.alt = "Source image";
    card.querySelector(".thumbnail-button").append(corner);
  }
}
function queueThumbnail(job) {
  if (
    !createViewer ||
    !job.preview_available ||
    job.thumbnail_available ||
    thumbURLs.has(job.job_id) ||
    thumbPending.has(job.job_id)
  )
    return;
  thumbPending.add(job.job_id);
  thumbQueue.push(job);
  makeThumbnails();
}
async function makeThumbnails() {
  if (thumbBusy || !createViewer) return;
  thumbBusy = true;
  try {
    if (!thumbViewer) {
      const host = node("div");
      host.id = "thumb-host";
      host.setAttribute("aria-hidden", "true");
      document.body.append(host);
      thumbViewer = createViewer(host, { thumbnail: true });
    }
    while (thumbQueue.length) {
      const job = thumbQueue.shift();
      try {
        await thumbViewer.load(
          previewURL(job),
          job.backend === "triposr" ? "z" : "y",
        );
        const blob = await thumbViewer.snapshot();
        await api(`/v1/jobs/${job.job_id}/thumbnail`, {
          method: "PUT",
          headers: { "Content-Type": "image/png" },
          body: blob,
        });
        const url = `/v1/jobs/${job.job_id}/thumbnail`;
        thumbURLs.set(job.job_id, url);
        applyThumbnail(job, url);
      } catch {
        /* A source thumbnail remains visible if a mesh cannot be rendered. */
      }
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  } catch {
    thumbQueue.length = 0;
  } finally {
    thumbBusy = false;
  }
}
function jobCard(job) {
  const signature = JSON.stringify([job, selectedJob === job.job_id]),
    previous = cards.get(job.job_id);
  if (previous?.signature === signature) return previous.node;
  const card = node(
    "article",
    `job ${job.status}${selectedJob === job.job_id ? " selected" : ""}`,
  );
  card.dataset.jobId = job.job_id;
  const thumb = button("", () => openPreview(job), "thumbnail-button");
  thumb.disabled = !job.preview_available;
  thumb.setAttribute(
    "aria-label",
    `Preview ${job.source_name || "mesh"} with ${names[job.backend]}`,
  );
  const image = node("img");
  image.src = job.thumbnail_available
    ? `/v1/jobs/${job.job_id}/thumbnail`
    : thumbURLs.get(job.job_id) || sourceURL(job);
  image.alt = job.thumbnail_available
    ? "Generated mesh"
    : job.source_name || "Source image";
  image.loading = "lazy";
  image.addEventListener("error", () => {
    image.hidden = true;
  });
  thumb.append(
    image,
    node(
      "span",
      "thumb-label",
      job.thumbnail_available || thumbURLs.has(job.job_id) ? "MESH" : "SOURCE",
    ),
  );
  card.append(thumb);
  const body = node("div", "job-body");
  const title = node(
    "p",
    "job-source",
    job.source_name || job.job_id.slice(0, 8),
  );
  title.title = title.textContent;
  body.append(title);
  const model = node("div", "job-title", names[job.backend] || job.backend);
  model.append(
    node(
      "span",
      `status ${job.status}`,
      job.cancellation_requested && !terminal.has(job.status)
        ? "Cancelling…"
        : job.status,
    ),
  );
  body.append(model);
  if (job.parent_job_id) body.append(node("p", "hint", `Refined from ${job.parent_job_id.slice(0, 8)}`));
  if (Object.keys(job.view_assets || {}).length) body.append(node("p", "hint", `Views: ${Object.keys(job.view_assets).join(", ")}`));
  const result = job.result?.backend;
  const bits = [];
  if (job.parameters.seed !== undefined)
    bits.push(`Seed ${job.parameters.seed}`);
  if (job.parameters.guidance_scale !== undefined)
    bits.push(`Guidance ${job.parameters.guidance_scale}`);
  if (job.parameters.density_threshold !== undefined)
    bits.push(`Threshold ${job.parameters.density_threshold}`);
  bits.push(
    `${job.parameters.octree_resolution || job.parameters.mc_resolution || 256} res`,
  );
  if (result?.faces)
    bits.push(`${Number(result.faces).toLocaleString()} faces`);
  if (job.result?.wall_seconds)
    bits.push(`${Math.round(job.result.wall_seconds)}s`);
  body.append(node("div", "job-meta", bits.join(" · ")));
  if (job.error) body.append(node("p", "job-error", job.error));
  const actions = node("div", "job-actions");
  if (job.status === "succeeded") {
    if (job.preview_available)
      actions.append(button("Inspect", () => openPreview(job)));
    actions.append(downloadLink(job));
    const refine = button("Refine with UltraShape", () => refineJob(job));
    refine.disabled = !refinementReady;
    refine.title = "Uses the UltraShape settings in the sidebar and this job's source image";
    actions.append(refine);
    if (job.parent_job_id) actions.append(button("View original", async () => {
      try { await openPreview(await api(`/v1/jobs/${job.parent_job_id}`)); }
      catch (error) { notice(error.message, true); }
    }));
  } else if (!terminal.has(job.status)) {
    const cancel = button("Cancel", () => cancelJob(job));
    cancel.disabled = job.cancellation_requested;
    actions.append(cancel);
  }
  actions.append(button("Reuse settings", () => reuseSettings(job)));
  body.append(actions);
  card.append(body);
  const details = node("details");
  if (previous?.node.querySelector("details")?.open) details.open = true;
  details.append(
    node("summary", "", "Run details"),
    node(
      "pre",
      "",
      JSON.stringify(
        { parameters: job.parameters, result: job.result },
        null,
        2,
      ),
    ),
  );
  card.append(details);
  cards.set(job.job_id, { node: card, signature });
  if (job.thumbnail_available || thumbURLs.has(job.job_id))
    applyThumbnail(
      job,
      job.thumbnail_available
        ? `/v1/jobs/${job.job_id}/thumbnail`
        : thumbURLs.get(job.job_id),
    );
  queueThumbnail(job);
  return card;
}
function renderGallery() {
  const renderSignature = JSON.stringify([
    groups,
    selectedJob,
    queueOnly,
    $("search").value,
    $("status-filter").value,
    [...collapsed],
  ]);
  if (renderSignature === lastRenderSignature) return;
  lastRenderSignature = renderSignature;
  const counts = statusCounts(jobs),
    active = (counts.running || 0) + (counts.queued || 0);
  $("queue-count").textContent = active;
  $("queue-bar").replaceChildren(
    node(
      "span",
      "",
      counts.running
        ? `● Generating · ${counts.queued || 0} queued`
        : `${counts.queued || 0} queued · Ready to create`,
    ),
    node(
      "span",
      "",
      `${counts.succeeded || 0} completed · ${counts.failed || 0} failed · sequential GPU processing`,
    ),
  );
  const fragment = document.createDocumentFragment();
  let visible = 0;
  for (const group of groups) {
    const filtered = group.jobs.filter((job) => selectedFilter(job, group));
    if (!filtered.length) continue;
    visible += filtered.length;
    const section = node("section", "batch");
    section.dataset.batchId = group.id;
    const header = node("div", "batch-header");
    const title = node("div");
    const toggle = button(
      "",
      () => {
        if (collapsed.has(group.id)) collapsed.delete(group.id);
        else collapsed.add(group.id);
        renderGallery();
      },
      "batch-title",
    );
    toggle.setAttribute("aria-expanded", !collapsed.has(group.id));
    const completed = group.jobs.filter((job) =>
      terminal.has(job.status),
    ).length;
    toggle.append(
      node(
        "strong",
        "",
        `${collapsed.has(group.id) ? "▸" : "▾"} ${group.name}`,
      ),
      node("span", "", `${completed} / ${group.jobs.length}`),
    );
    title.append(
      toggle,
      node(
        "p",
        "batch-meta",
        `${new Set(group.jobs.flatMap((job) => [job.asset_id, ...Object.values(job.view_assets || {})])).size} source images · ${new Date(group.created_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`,
      ),
    );
    header.append(title);
    const actions = node("div", "batch-actions");
    if (
      group.id !== "legacy" &&
      group.jobs.some((job) => job.status === "succeeded")
    ) {
      const download = node("a", "download", "Download batch ↓");
      download.href = `/v1/batches/${group.id}/artifacts`;
      download.download = `meshtitan-${group.id}.zip`;
      actions.append(download);
    }
    if (
      group.id !== "legacy" &&
      group.jobs.some((job) => !terminal.has(job.status))
    )
      actions.append(button("Cancel batch", () => cancelBatch(group)));
    header.append(actions);
    section.append(header);
    if (completed < group.jobs.length) {
      const progress = node("div", "batch-progress");
      const bar = node("span");
      bar.style.width = `${(completed / group.jobs.length) * 100}%`;
      progress.append(bar);
      section.append(progress);
    }
    const grid = node("div", "batch-grid");
    grid.hidden = collapsed.has(group.id);
    for (const job of filtered) grid.append(jobCard(job));
    section.append(grid);
    fragment.append(section);
  }
  $("groups").replaceChildren(fragment);
  $("history-empty").hidden = visible > 0;
  $("history-empty").textContent = jobs.length
    ? "No generations match these filters."
    : "Your batches will appear here. Import images, choose your models, and generate.";
  $("load-more").hidden = groupLimit >= groupTotal;
}
async function refreshGallery() {
  if (refreshing) return;
  refreshing = true;
  try {
    const fetched = [];
    for (let offset = 0; offset < groupLimit; offset += 50) {
      const data = await api(
        `/v1/gallery?limit=${Math.min(groupLimit - offset, 50)}&offset=${offset}`,
      );
      fetched.push(...data.groups);
      groupTotal = data.total;
      if (fetched.length >= groupTotal) break;
    }
    groups = fetched;
    jobs = groups.flatMap((group) => group.jobs);
    renderGallery();
    return true;
  } catch (error) {
    notice(
      `Cannot refresh gallery: ${error.message}. Retrying automatically.`,
      true,
    );
    return false;
  } finally {
    refreshing = false;
  }
}
async function openPreview(
  job,
  upAxis = job.backend === "triposr" ? "z" : "y",
) {
  selectedJob = job.job_id;
  renderGallery();
  $("up-axis").value = upAxis;
  $("preview-title").textContent =
    `${job.source_name || "Mesh"} · ${names[job.backend]}`;
  $("preview-badge").textContent =
    `${job.batch_name || "GENERATION"} · ${job.output_format.toUpperCase()}`;
  $("preview-download").href = artifactURL(job);
  $("preview-download").download =
    `${job.backend}-${job.job_id.slice(0, 8)}.${job.output_format}`;
  $("preview-download").textContent =
    `Download ${job.output_format.toUpperCase()} ↓`;
  const result = job.result?.backend;
  $("mesh-stats").textContent = result
    ? `${Number(result.vertices).toLocaleString()} vertices · ${Number(result.faces).toLocaleString()} faces · ${(job.result.mesh_bytes / 1048576).toFixed(1)} MB`
    : "Mesh preview";
  $("inspector-details").textContent = JSON.stringify(job, null, 2);
  $("inspector-actions").replaceChildren(
    button("Reuse settings", () => {
      reuseSettings(job);
      $("preview-dialog").close();
    }),
    button("Add source to inputs", () => reuseSource(job)),
  );
  if (!$("preview-dialog").open) $("preview-dialog").showModal();
  $("viewer-message").replaceChildren(node("h3", "", "Loading mesh…"));
  $("viewer-message").hidden = false;
  try {
    if (!viewer)
      throw new Error("WebGL is unavailable. You can still download the mesh.");
    if (
      (await viewer.load(previewURL(job), upAxis)) &&
      selectedJob === job.job_id
    ) {
      viewer.setColor($("mesh-color").value);
      $("viewer-message").hidden = true;
    }
  } catch (error) {
    $("viewer-message").replaceChildren(
      node("h3", "", "Preview unavailable"),
      node("p", "", error.message),
    );
  }
}
async function connect() {
  try {
    const data = await api("/v1/capabilities");
    $("device").textContent =
      `${data.gpu.name} · ${Math.round(data.gpu.memory_total_mib / 1024)} GB`;
    uploadLimit = data.max_upload_mib * 1048576;
    $("image-caption").textContent =
      `PNG, JPG or WebP · ${data.max_upload_mib} MB each`;
    for (const name of generators) {
      const available = data.backends[name]?.enabled && data.backends[name]?.ok;
      $(`select-${name}`).disabled = !available;
      if (!available) $(`select-${name}`).checked = false;
    }
    refinementReady = Boolean(data.backends.ultrashape?.enabled && data.backends.ultrashape?.ok);
    multiviewReady = Boolean(data.backends.hunyuan_multiview?.enabled && data.backends.hunyuan_multiview?.ok);
    $("refinement-status").textContent = refinementReady ? "Ready · settings below apply to refinement buttons." : "Unavailable · run windows.bat setup to install UltraShape.";
    $("params-ultrashape").disabled = !refinementReady;
    ready = true;
  } catch (error) {
    ready = false;
    $("device").textContent = "Runtime unavailable";
    notice(error.message, true);
  }
  updateForm();
}
$("search").addEventListener("input", renderGallery);
$("status-filter").addEventListener("change", renderGallery);
$("grid-size").addEventListener("input", () =>
  document.documentElement.style.setProperty(
    "--tile-size",
    `${$("grid-size").value}px`,
  ),
);
function setTab(queued) {
  queueOnly = queued;
  for (const [id, active] of [
    ["tab-all", !queued],
    ["tab-queue", queued],
  ]) {
    $(id).classList.toggle("active", active);
    $(id).setAttribute("aria-pressed", active);
  }
  renderGallery();
}
$("tab-all").addEventListener("click", () => setTab(false));
$("tab-queue").addEventListener("click", () => setTab(true));
$("refresh").addEventListener("click", async () => {
  await connect();
  await refreshGallery();
});
$("load-more").addEventListener("click", () => {
  groupLimit += 20;
  refreshGallery();
});
$("close-preview").addEventListener("click", () => $("preview-dialog").close());
$("fit-view").addEventListener("click", () => viewer?.fit());
$("wireframe").addEventListener("change", (event) =>
  viewer?.setWireframe(event.target.checked),
);
$("auto-rotate").addEventListener("change", (event) =>
  viewer?.setRotate(event.target.checked),
);
$("mesh-color").addEventListener("input", (event) =>
  viewer?.setColor(event.target.value),
);
$("up-axis").addEventListener("change", (event) => {
  const job = jobs.find((item) => item.job_id === selectedJob);
  if (job) openPreview(job, event.target.value);
});
try {
  ({ createViewer } = await import("./viewer.js"));
  viewer = createViewer($("canvas-host"));
} catch {
  createViewer = null;
}
await connect();
await refreshGallery();
setInterval(async () => {
  if (!ready) await connect();
  if (!document.hidden) await refreshGallery();
}, 3000);
updateForm();
