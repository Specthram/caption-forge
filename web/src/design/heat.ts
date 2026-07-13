/**
 * Heat-map rendering for SigLIP grounding.
 *
 * The backend hands each element a `side * side` grid of bytes, one per
 * image patch, row-major (see `src.siglip_grounding._heat_grid`): 255 is the
 * patch that supports the text most, 0 the least. Everything here turns that
 * into pixels.
 *
 * The palette is **Turbo** — Google's perceptually uniform rainbow, the
 * classic thermal ramp (deep blue → cyan → green → yellow → red). Unlike
 * `jet` it has no false detail bands, and unlike a single-hue ramp it reads
 * as temperature at a glance, which is exactly what "where does this claim
 * live in the picture" wants. Cold patches fade to transparent rather than
 * to blue, so the image stays visible underneath.
 */

/** Byte value a patch must reach to count as covered by an element. */
export const COVERAGE_CUT = 140;

/**
 * How sharply cold patches fade out. Above 1 the low end is pushed toward
 * transparent, so only the genuinely supporting region is painted; at 1 the
 * whole frame would carry a wash of blue. Kept low enough that warm patches
 * stay clearly opaque — a value near the top of the grid must read as hot,
 * not faint.
 */
const ALPHA_GAMMA = 1.4;

/** Turbo colormap, Mikhailov's polynomial fit. `t` is clamped to [0, 1]. */
export function turbo(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  const r =
    34.61 +
    x * (1172.33 + x * (-10793.56 + x * (33300.12 + x * (-38394.49 + x * 14825.05))));
  const g =
    23.31 + x * (557.33 + x * (1225.33 + x * (-3574.96 + x * (1073.77 + x * 707.56))));
  const b =
    27.2 + x * (3211.1 + x * (-15327.97 + x * (27814 + x * (-22569.18 + x * 6838.66))));
  return [clamp255(r), clamp255(g), clamp255(b)];
}

function clamp255(value: number): number {
  return Math.min(255, Math.max(0, Math.round(value)));
}

/** Decode a base64 heat grid into its raw bytes (one per patch). */
export function decodeHeat(encoded: string): Uint8Array {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/**
 * The pixelwise maximum of several grids — the union of what they support.
 *
 * This is what makes the modal's "Couverture" a real measurement rather than
 * the prototype's hand-authored blob areas: overlapping claims count once,
 * because a patch is covered by the *strongest* claim that reaches it.
 */
export function unionGrid(grids: Uint8Array[]): Uint8Array | null {
  if (grids.length === 0) return null;
  const union = new Uint8Array(grids[0].length);
  for (const grid of grids) {
    for (let i = 0; i < union.length; i += 1) {
      if (grid[i] > union[i]) union[i] = grid[i];
    }
  }
  return union;
}

/** The share of image area a grid covers, as a 0-100 percentage. */
export function coveragePct(grid: Uint8Array | null): number {
  if (!grid || grid.length === 0) return 0;
  let covered = 0;
  for (const value of grid) if (value >= COVERAGE_CUT) covered += 1;
  return Math.round((100 * covered) / grid.length);
}

/**
 * Paint a heat grid onto a canvas, upscaled to its display size.
 *
 * The grid is written pixel-per-patch into a tiny offscreen buffer, then
 * drawn scaled with smoothing on — the browser's bilinear filter is both
 * faster and smoother than interpolating in JS. Callers soften the patch
 * grid further with a CSS `blur()` and composite with `mix-blend-mode:
 * screen`, which is why the alpha channel (not a black background) carries
 * the "no evidence here" signal.
 *
 * @param strength Global opacity multiplier, 0-1. The modal scales it by the
 *   element's own score, so a hallucinated claim stays dim even though its
 *   brightest patch normalizes to 255.
 */
export function paintHeat(
  canvas: HTMLCanvasElement,
  grid: Uint8Array,
  side: number,
  strength: number,
): void {
  const context = canvas.getContext("2d");
  if (!context) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (strength <= 0) return;

  const buffer = document.createElement("canvas");
  buffer.width = side;
  buffer.height = side;
  const bufferContext = buffer.getContext("2d");
  if (!bufferContext) return;

  const image = bufferContext.createImageData(side, side);
  for (let i = 0; i < grid.length; i += 1) {
    const value = grid[i] / 255;
    const [r, g, b] = turbo(value);
    image.data[i * 4] = r;
    image.data[i * 4 + 1] = g;
    image.data[i * 4 + 2] = b;
    image.data[i * 4 + 3] = clamp255(255 * strength * value ** ALPHA_GAMMA);
  }
  bufferContext.putImageData(image, 0, 0);

  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(buffer, 0, 0, canvas.width, canvas.height);
}
