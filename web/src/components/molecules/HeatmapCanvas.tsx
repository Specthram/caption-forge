/**
 * The thermal overlay of the grounding modal.
 *
 * One canvas laid over the image, repainted whenever the shown grid changes.
 * The grid is `side * side` bytes (one per SigLIP patch), so it is upscaled
 * to the frame with the browser's bilinear filter and softened with a CSS
 * blur — a patch grid is coarse (16x16 at 256px, 32x32 at 512px) and hard
 * patch edges would read as false structure.
 *
 * `mix-blend-mode: screen` keeps the picture legible underneath: cold
 * patches are transparent and warm ones brighten rather than paint over.
 *
 * **Alignment.** SigLIP's processor squashes the image to a square without
 * preserving its aspect ratio, so patch `(row, col)` maps linearly onto the
 * *whole* original image. The canvas must therefore cover exactly the
 * rendered image box — the caller wraps image and canvas in a shrink-to-fit
 * container and shows the image with `object-fit: contain`. Cropping it
 * (`cover`) would slide the heat off the pixels it describes.
 */

import { useEffect, useRef } from "react";
import { paintHeat } from "../../design/heat";

/** Canvas backing-store resolution. Enough to keep the blur smooth. */
const RESOLUTION = 512;

interface HeatmapCanvasProps {
  /** The grid to paint, or null to clear the overlay. */
  grid: Uint8Array | null;
  /** Patches per side of that grid. */
  side: number;
  /** Global opacity, 0-1 (the modal scales it by the element's score). */
  strength: number;
  /** Blur radius in px, applied in CSS over the upscaled grid. */
  blur?: number;
}

export function HeatmapCanvas({
  grid,
  side,
  strength,
  blur = 10,
}: HeatmapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!grid || side <= 0) {
      canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    paintHeat(canvas, grid, side, strength);
  }, [grid, side, strength]);

  return (
    <canvas
      ref={canvasRef}
      width={RESOLUTION}
      height={RESOLUTION}
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        filter: `blur(${blur}px)`,
        mixBlendMode: "screen",
        transition: "opacity 0.2s ease",
        opacity: grid ? 1 : 0,
      }}
    />
  );
}
