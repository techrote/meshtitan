export function guidanceValues(
  base,
  { enabled = false, count = 1, step = 1 } = {},
) {
  if (!enabled) return [base];
  if (
    !Number.isInteger(count) ||
    count < 1 ||
    count > 16 ||
    !Number.isFinite(step)
  ) {
    throw new Error("Guidance sweep needs 1–16 values and a finite increment.");
  }
  const values = Array.from({ length: count }, (_, i) =>
    Number((base + i * step).toFixed(6)),
  );
  if (
    values.some((value) => !Number.isFinite(value) || value < 0 || value > 30)
  ) {
    throw new Error(
      "Every guidance value must be between 0 and 30. Adjust the start, increment or count.",
    );
  }
  return values;
}

export function planBatch({
  assets,
  backends,
  settings,
  variations,
  seedMode,
  sweep,
  randomSeed,
  multiview = false,
}) {
  if (!Number.isInteger(variations) || variations < 1 || variations > 8) {
    throw new Error("Choose 1–8 seed variations.");
  }
  const hunyuan = backends.find(name => name.startsWith("hunyuan_"));
  const guidance = hunyuan
    ? guidanceValues(settings[hunyuan].guidance_scale, sweep)
    : [];
  if (multiview) {
    const views = Object.fromEntries(assets.map(asset => [asset.view, asset.asset_id]));
    if (assets.length < 2 || assets.length > 4 || Object.keys(views).length !== assets.length ||
        !views.front || Object.keys(views).some(view => !["front", "left", "back", "right"].includes(view)) ||
        backends.length !== 1 || backends[0] !== "hunyuan_multiview")
      throw new Error("Multi-view requires unique labelled views including front, and the multi-view backend.");
    const front = assets.find(asset => asset.view === "front");
    assets = [{asset_id: front.asset_id, source_name: front.source_name,
      output_format: front.output_format, view_assets: views}];
  }
  const requests = [];
  let variationIndex = 0;
  for (const asset of assets) {
    for (const backend of backends) {
      if (backend.startsWith("hunyuan_")) {
        for (let i = 0; i < variations; i++) {
          const base = settings[backend];
          // A seed is chosen once per variation, then held fixed across the guidance sweep.
          const seed =
            seedMode === "random"
              ? randomSeed()
              : seedMode === "increment"
                ? (base.seed + variationIndex) % 4294967296
                : base.seed;
          variationIndex++;
          for (const strength of guidance)
            requests.push({
              backend,
              asset_id: asset.asset_id,
              source_name: asset.source_name,
              output_format: asset.output_format,
              parameters: { ...base, seed, guidance_scale: strength },
              ...(asset.view_assets ? {view_assets: asset.view_assets} : {}),
            });
        }
      } else
        requests.push({
          backend,
          asset_id: asset.asset_id,
          source_name: asset.source_name,
          output_format: asset.output_format,
          parameters: { ...settings[backend] },
        });
    }
  }
  if (!requests.length || requests.length > 200)
    throw new Error("A batch must contain 1–200 jobs.");
  return requests;
}
