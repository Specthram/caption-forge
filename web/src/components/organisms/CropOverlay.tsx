/**
 * Crop overlay — frame a rectangle on an image with pixel precision.
 *
 * The rectangle is stored in percentages of the *source* image, so it never
 * depends on how the image is displayed. The viewport owns a separate
 * pan/zoom transform (`zoom`) used to inspect the image while framing: it
 * scales the pixels, never the rectangle, so every drag delta is divided by
 * `z` and every affordance (handles, border, size label) is counter-scaled by
 * `1/z` to keep a constant size on screen.
 *
 * Validating a NEW rectangle opens the replace/beside dialog: a crop is a
 * virtual alias, so the user chooses whether it takes the original's place in
 * the dataset or stands beside it as a second sample. Re-framing an existing
 * crop saves straight away.
 */

import { useEffect, useRef } from "react";
import {
  useCreateCrop,
  useCropSource,
  useDatasets,
  useUpdateCrop,
} from "../../api/hooks";
import type { CropRatio, CropRect } from "../../api/types";
import { colors, font } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";

/** Smallest crop side, in percent of the source (mirrors src.crops.MIN_SIDE). */
const MIN = 8;

const RATIOS: CropRatio[] = [
  "free",
  "1:1",
  "3:2",
  "2:3",
  "4:3",
  "3:4",
  "16:9",
  "9:16",
];

/** The aspect ratios a trainer buckets images into, for the deploy readout. */
const BUCKETS: { ratio: number; label: string }[] = [
  { ratio: 1, label: "1:1" },
  { ratio: 3 / 2, label: "3:2" },
  { ratio: 2 / 3, label: "2:3" },
  { ratio: 4 / 3, label: "4:3" },
  { ratio: 3 / 4, label: "3:4" },
  { ratio: 16 / 9, label: "16:9" },
  { ratio: 9 / 16, label: "9:16" },
  { ratio: 7 / 5, label: "7:5" },
  { ratio: 5 / 7, label: "5:7" },
];

type Handle = "nw" | "ne" | "se" | "sw" | "n" | "s" | "e" | "w";

const CORNERS: Handle[] = ["nw", "ne", "se", "sw"];
const EDGES: Handle[] = ["n", "s", "e", "w"];

const HANDLE_POS: Record<Handle, { top?: string; left?: string; cursor: string }> =
  {
    nw: { top: "0%", left: "0%", cursor: "nwse-resize" },
    ne: { top: "0%", left: "100%", cursor: "nesw-resize" },
    se: { top: "100%", left: "100%", cursor: "nwse-resize" },
    sw: { top: "100%", left: "0%", cursor: "nesw-resize" },
    n: { top: "0%", left: "50%", cursor: "ns-resize" },
    s: { top: "100%", left: "50%", cursor: "ns-resize" },
    e: { top: "50%", left: "100%", cursor: "ew-resize" },
    w: { top: "50%", left: "0%", cursor: "ew-resize" },
  };

const clamp = (value: number, lo: number, hi: number) =>
  Math.max(lo, Math.min(hi, value));

/** The numeric aspect ratio of a lock, or null when the frame is free. */
function ratioNum(ratio: CropRatio): number | null {
  if (ratio === "free") return null;
  const [w, h] = ratio.split(":").map(Number);
  return w / h;
}

/**
 * The crop's pixel size, and the size it deploys at after the resize.
 *
 * The sides are the *difference of the rounded edges*, exactly how the
 * backend resolves the rectangle (`src.crops.pixel_rect`) — rounding the
 * width directly would read one pixel off what actually gets rendered.
 */
function effectiveRes(rect: CropRect, srcW: number, srcH: number, res: number) {
  const left = Math.round((rect.x / 100) * srcW);
  const top = Math.round((rect.y / 100) * srcH);
  const cw = Math.max(1, Math.round(((rect.x + rect.w) / 100) * srcW) - left);
  const ch = Math.max(1, Math.round(((rect.y + rect.h) / 100) * srcH) - top);
  const short = Math.min(cw, ch);
  if (!res || short <= res) return { cw, ch, ow: cw, oh: ch };
  const scale = res / short;
  return { cw, ch, ow: Math.round(cw * scale), oh: Math.round(ch * scale) };
}

/** The training bucket a size lands in; `ok` when within 2% of it. */
function bucketOf(cw: number, ch: number) {
  const aspect = cw / ch;
  let best = BUCKETS[0];
  let distance = Infinity;
  for (const bucket of BUCKETS) {
    const delta = Math.abs(bucket.ratio - aspect) / bucket.ratio;
    if (delta < distance) {
      distance = delta;
      best = bucket;
    }
  }
  return { label: best.label, ok: distance <= 0.02 };
}

/**
 * Re-derive the height from the width under a ratio lock, keeping the
 * rectangle inside the image (shrinking the width when it would overflow).
 */
function applyRatio(
  rect: CropRect,
  ratio: CropRatio,
  aspect: number,
): CropRect {
  const target = ratioNum(ratio);
  if (!target) return rect;
  const next = { ...rect };
  let height = (next.w * aspect) / target;
  if (next.y + height > 100) {
    height = 100 - next.y;
    next.w = (height * target) / aspect;
  }
  next.h = height;
  return next;
}

interface DragState {
  handle: Handle | "move" | "pan";
  sx: number;
  sy: number;
  rect: CropRect;
  bw: number;
  bh: number;
  z: number;
  tx: number;
  ty: number;
}

export function CropOverlay() {
  const crop = useUiStore((state) => state.crop);
  const datasetId = useUiStore((state) => state.datasetId);
  const setCrop = useUiStore((state) => state.setCrop);
  const close = useUiStore((state) => state.closeCrop);

  const source = useCropSource(crop.open ? crop.mediaId : null);
  const datasets = useDatasets();
  const createCrop = useCreateCrop();
  const updateCrop = useUpdateCrop();

  const boxRef = useRef<HTMLDivElement>(null);
  const drag = useRef<DragState | null>(null);

  // Escape closes the confirm dialog first, then the overlay — the same
  // "one step back" the ghost link offers.
  useEffect(() => {
    if (!crop.open) return undefined;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (useUiStore.getState().crop.confirm) setCrop({ confirm: false });
      else close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [crop.open, close, setCrop]);

  if (!crop.open || crop.mediaId == null || !source.data) return null;

  const { width: srcW, height: srcH, name, file } = source.data;
  const aspect = srcW / srcH;
  const dataset = datasets.data?.datasets.find((row) => row.id === datasetId);
  const resolution = dataset?.deploy_resolution ?? 0;
  const { cw, ch, ow, oh } = effectiveRes(crop.rect, srcW, srcH, resolution);
  const bucket = bucketOf(ow, oh);
  const { z, tx, ty } = crop.zoom;
  const editing = crop.editId != null;

  const onDown = (
    event: React.MouseEvent,
    handle: Handle | "move" | "pan",
  ) => {
    event.preventDefault();
    event.stopPropagation();
    const box = boxRef.current?.getBoundingClientRect();
    if (!box) return;
    drag.current = {
      handle,
      sx: event.clientX,
      sy: event.clientY,
      rect: { ...crop.rect },
      bw: box.width,
      bh: box.height,
      z,
      tx,
      ty,
    };
    const move = (moveEvent: MouseEvent) => onDrag(moveEvent);
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      drag.current = null;
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };

  const onDrag = (event: MouseEvent) => {
    const state = drag.current;
    if (!state) return;
    if (state.handle === "pan") {
      setCrop({
        zoom: {
          z: state.z,
          tx: clamp(
            state.tx + (event.clientX - state.sx),
            state.bw * (1 - state.z),
            0,
          ),
          ty: clamp(
            state.ty + (event.clientY - state.sy),
            state.bh * (1 - state.z),
            0,
          ),
        },
      });
      return;
    }
    // Deltas are in *source* percent: divide by the viewport zoom so the
    // rectangle tracks the cursor 1:1 whatever the magnification.
    const dx = ((event.clientX - state.sx) / (state.bw * state.z)) * 100;
    const dy = ((event.clientY - state.sy) / (state.bh * state.z)) * 100;
    const { x, y } = state.rect;
    let { w, h } = state.rect;

    if (state.handle === "move") {
      setCrop({
        rect: {
          x: clamp(x + dx, 0, 100 - w),
          y: clamp(y + dy, 0, 100 - h),
          w,
          h,
        },
      });
      return;
    }

    const lock = ratioNum(crop.ratio);
    const ex = state.handle.includes("w") ? -1 : state.handle.includes("e") ? 1 : 0;
    const ey = state.handle.includes("n") ? -1 : state.handle.includes("s") ? 1 : 0;
    let left = x;
    let top = y;
    let right = x + w;
    let bottom = y + h;
    if (ex < 0) left = clamp(left + dx, 0, right - MIN);
    if (ex > 0) right = clamp(right + dx, left + MIN, 100);
    if (ey < 0) top = clamp(top + dy, 0, bottom - MIN);
    if (ey > 0) bottom = clamp(bottom + dy, top + MIN, 100);
    w = right - left;
    h = bottom - top;

    if (lock && ex !== 0) {
      // Under a lock only the corners are live: derive the height from the
      // width, anchored to the corner opposite the one being dragged, and
      // re-derive the width whenever the height would leave the image.
      let height = (w * aspect) / lock;
      if (ey < 0) {
        top = bottom - height;
        if (top < 0) {
          top = 0;
          height = bottom;
          w = (height * lock) / aspect;
          left = right - w;
        }
      } else {
        bottom = top + height;
        if (bottom > 100) {
          bottom = 100;
          height = bottom - top;
          w = (height * lock) / aspect;
          left = right - w;
        }
      }
      if (left < 0) {
        left = 0;
        w = right - left;
        height = (w * aspect) / lock;
        if (ey < 0) top = bottom - height;
        else bottom = top + height;
      }
      h = bottom - top;
    }
    setCrop({ rect: { x: left, y: top, w, h } });
  };

  const onWheel = (event: React.WheelEvent) => {
    const box = boxRef.current?.getBoundingClientRect();
    if (!box) return;
    const sx = event.clientX - box.left;
    const sy = event.clientY - box.top;
    const next = clamp(z * (event.deltaY > 0 ? 0.9 : 1.1), 1, 10);
    setCrop({
      zoom: {
        z: next,
        // Keep the point under the cursor fixed while scaling.
        tx: clamp(sx - ((sx - tx) / z) * next, box.width * (1 - next), 0),
        ty: clamp(sy - ((sy - ty) / z) * next, box.height * (1 - next), 0),
      },
    });
  };

  const setPixel = (field: "x" | "y" | "w" | "h", raw: string) => {
    const value = parseInt(raw, 10);
    if (Number.isNaN(value)) return;
    const lock = ratioNum(crop.ratio);
    let { x, y, w, h } = crop.rect;
    if (field === "w") {
      w = clamp((value / srcW) * 100, 1, 100);
      if (lock) h = (w * aspect) / lock;
    } else if (field === "h") {
      h = clamp((value / srcH) * 100, 1, 100);
      if (lock) w = (h * lock) / aspect;
    } else if (field === "x") {
      x = clamp((value / srcW) * 100, 0, 100);
    } else {
      y = clamp((value / srcH) * 100, 0, 100);
    }
    if (x + w > 100) x = Math.max(0, 100 - w);
    if (y + h > 100) y = Math.max(0, 100 - h);
    setCrop({ rect: { x, y, w, h } });
  };

  const validate = () => {
    if (crop.editId != null) {
      updateCrop.mutate(
        { id: crop.editId, rect: crop.rect, ratio: crop.ratio },
        { onSuccess: close },
      );
      return;
    }
    setCrop({ confirm: true });
  };

  const place = (mode: "replace" | "beside") => {
    if (crop.mediaId == null || datasetId == null) return;
    createCrop.mutate(
      {
        media_id: crop.mediaId,
        rect: crop.rect,
        ratio: crop.ratio,
        dataset_id: datasetId,
        mode,
      },
      { onSuccess: close },
    );
  };

  const numberInput = (field: "x" | "y" | "w" | "h", value: number) => (
    <input
      type="number"
      value={value}
      onChange={(event) => setPixel(field, event.target.value)}
      style={{
        width: field === "x" || field === "y" ? 62 : 66,
        background: colors.input,
        border: `1px solid ${colors.borderControl}`,
        borderRadius: 5,
        color: colors.text,
        fontFamily: font.mono,
        fontSize: 11,
        padding: "4px 6px",
      }}
    />
  );

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 78,
        background: "rgba(8,9,11,0.9)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: 22,
        gap: 14,
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          width: "100%",
          maxWidth: 1020,
        }}
      >
        <span style={{ fontSize: 13.5, fontWeight: 700, color: colors.text }}>
          ⌗ Crop — {name}
        </span>
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 10.5,
            color: colors.textMuted,
          }}
        >
          {editing ? "editing" : `new · ${dataset?.name ?? "no dataset"}`}
        </span>
        <span style={{ flex: 1 }} />
        <button
          type="button"
          onClick={close}
          title="Close"
          style={{
            width: 26,
            height: 26,
            borderRadius: 6,
            background: colors.card,
            border: `1px solid ${colors.borderControl}`,
            color: colors.textMutedAlt,
            cursor: "pointer",
          }}
        >
          ✕
        </button>
      </header>

      <div
        style={{
          flex: 1,
          width: "100%",
          maxWidth: 1020,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          minHeight: 0,
        }}
      >
        <div
          ref={boxRef}
          onWheel={onWheel}
          style={{
            position: "relative",
            aspectRatio: `${srcW} / ${srcH}`,
            maxWidth: "100%",
            height: "min(60vh, 100%)",
            borderRadius: 4,
            overflow: "hidden",
            userSelect: "none",
          }}
        >
          <div
            onMouseDown={(event) => onDown(event, "pan")}
            style={{
              position: "absolute",
              inset: 0,
              transformOrigin: "0 0",
              transform: `translate(${tx}px, ${ty}px) scale(${z})`,
              backgroundImage: `url(${file})`,
              backgroundSize: "100% 100%",
              cursor: z > 1 ? "grab" : "default",
            }}
          >
            <div
              onMouseDown={(event) => onDown(event, "move")}
              style={{
                position: "absolute",
                left: `${crop.rect.x}%`,
                top: `${crop.rect.y}%`,
                width: `${crop.rect.w}%`,
                height: `${crop.rect.h}%`,
                boxShadow: "0 0 0 9999px rgba(10,11,14,0.64)",
                border: `${1.4 / z}px solid ${colors.accent}`,
                cursor: "move",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  pointerEvents: "none",
                  backgroundImage:
                    "linear-gradient(to right, rgba(232,147,90,0.32) 1px, transparent 1px)," +
                    "linear-gradient(to bottom, rgba(232,147,90,0.32) 1px, transparent 1px)",
                  backgroundSize: "33.333% 33.333%",
                }}
              />
              <span
                style={{
                  position: "absolute",
                  left: "50%",
                  top: "50%",
                  transform: `translate(-50%, -50%) scale(${1 / z})`,
                  pointerEvents: "none",
                  fontFamily: font.mono,
                  fontSize: 11.5,
                  color: "#fff",
                  background: "rgba(10,11,14,0.72)",
                  borderRadius: 5,
                  padding: "2px 6px",
                  whiteSpace: "nowrap",
                }}
              >
                {cw}×{ch} px
              </span>
              {[...CORNERS, ...(crop.ratio === "free" ? EDGES : [])].map(
                (handle) => (
                  <span
                    key={handle}
                    onMouseDown={(event) => onDown(event, handle)}
                    style={{
                      position: "absolute",
                      top: HANDLE_POS[handle].top,
                      left: HANDLE_POS[handle].left,
                      width: 12,
                      height: 12,
                      margin: -6,
                      borderRadius: 3,
                      background: colors.accent,
                      cursor: HANDLE_POS[handle].cursor,
                      transform: `scale(${1 / z})`,
                    }}
                  />
                ),
              )}
            </div>
          </div>
          <span
            style={{
              position: "absolute",
              top: 6,
              left: 8,
              fontFamily: font.mono,
              fontSize: 10,
              color: "rgba(255,255,255,0.5)",
              pointerEvents: "none",
            }}
          >
            {name} · {srcW}×{srcH} · zoom {Math.round(z * 100)}%
          </span>
          {z > 1.001 && (
            <button
              type="button"
              onClick={() => setCrop({ zoom: { z: 1, tx: 0, ty: 0 } })}
              style={{
                position: "absolute",
                top: 6,
                right: 8,
                fontFamily: font.mono,
                fontSize: 10,
                color: colors.accent,
                background: "rgba(20,21,25,0.92)",
                border: `1px solid ${colors.borderControl}`,
                borderRadius: 5,
                padding: "2px 6px",
                cursor: "pointer",
              }}
            >
              1:1 ⟲
            </button>
          )}
        </div>
      </div>

      <div
        style={{
          width: "100%",
          maxWidth: 1020,
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              fontSize: 10,
              fontWeight: 600,
              textTransform: "uppercase",
              color: colors.textMuted,
            }}
          >
            Frame px
          </span>
          {numberInput("x", Math.round((crop.rect.x / 100) * srcW))}
          {numberInput("y", Math.round((crop.rect.y / 100) * srcH))}
          <span style={{ color: colors.textFaint }}>|</span>
          {numberInput("w", cw)}
          {numberInput("h", ch)}
          <span style={{ flex: 1 }} />
          <span
            style={{
              fontFamily: font.mono,
              fontSize: 10,
              color: colors.textFaint,
            }}
          >
            wheel: zoom · drag: pan · W/H follow the ratio
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span
            style={{
              fontSize: 10,
              fontWeight: 600,
              textTransform: "uppercase",
              color: colors.textMuted,
            }}
          >
            Ratio
          </span>
          <div
            style={{
              display: "inline-flex",
              border: `1px solid ${colors.borderControl}`,
              borderRadius: 6,
              overflow: "hidden",
            }}
          >
            {RATIOS.map((ratio) => {
              const active = ratio === crop.ratio;
              return (
                <button
                  key={ratio}
                  type="button"
                  onClick={() =>
                    setCrop({
                      ratio,
                      rect: applyRatio(crop.rect, ratio, aspect),
                    })
                  }
                  style={{
                    padding: "5px 10px",
                    border: "none",
                    fontFamily: font.mono,
                    fontSize: 10.5,
                    cursor: "pointer",
                    background: active ? colors.accentTint : "transparent",
                    color: active ? colors.accent : colors.textMutedAlt,
                  }}
                >
                  {ratio}
                </button>
              );
            })}
          </div>
          <span style={{ flex: 1 }} />
          <div style={{ textAlign: "right", fontFamily: font.mono }}>
            <div style={{ fontSize: 12, color: colors.text }}>
              crop {cw}×{ch}
            </div>
            <div style={{ fontSize: 10, color: colors.textMuted }}>
              → deploy {ow}×{oh} · bucket{" "}
              <span style={{ color: bucket.ok ? colors.ok : colors.warn }}>
                {bucket.label} {bucket.ok ? "✓" : "≈"}
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={close}
            style={{
              padding: "8px 14px",
              borderRadius: 7,
              border: `1px solid ${colors.borderControl}`,
              background: colors.card,
              color: colors.textMutedAlt,
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={validate}
            disabled={updateCrop.isPending}
            style={{
              padding: "8px 18px",
              borderRadius: 7,
              border: "none",
              background: colors.accent,
              color: colors.onAccent,
              fontSize: 12,
              fontWeight: 700,
              cursor: "pointer",
            }}
          >
            {editing ? "Save" : "Confirm →"}
          </button>
        </div>
      </div>

      {crop.confirm && (
        <ConfirmDialog
          name={name}
          size={`${ow}×${oh}`}
          pending={createCrop.isPending}
          onBack={() => setCrop({ confirm: false })}
          onChoose={place}
        />
      )}
    </div>
  );
}

/** The replace/beside choice shown when a NEW rectangle is validated. */
function ConfirmDialog({
  name,
  size,
  pending,
  onBack,
  onChoose,
}: {
  name: string;
  size: string;
  pending: boolean;
  onBack: () => void;
  onChoose: (mode: "replace" | "beside") => void;
}) {
  const option = (
    mode: "replace" | "beside",
    title: string,
    body: string,
    accent: boolean,
  ) => (
    <button
      type="button"
      disabled={pending}
      onClick={() => onChoose(mode)}
      style={{
        display: "block",
        textAlign: "left",
        padding: "10px 12px",
        borderRadius: 8,
        cursor: pending ? "wait" : "pointer",
        border: `1px solid ${accent ? colors.accentBorder : colors.borderControl}`,
        background: accent ? colors.accentTintAlt : colors.input,
      }}
    >
      <div
        style={{
          fontSize: 12.5,
          fontWeight: 600,
          color: accent ? colors.accent : colors.text,
        }}
      >
        {title}
      </div>
      <div style={{ fontSize: 11, color: colors.textMuted, marginTop: 2 }}>
        {body}
      </div>
    </button>
  );

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(8,9,11,0.6)",
      }}
    >
      <div
        style={{
          width: 440,
          background: colors.card,
          border: `1px solid ${colors.borderHover}`,
          borderRadius: 12,
          padding: 18,
          display: "flex",
          flexDirection: "column",
          gap: 14,
          boxShadow: "0 24px 80px rgba(0,0,0,0.6)",
        }}
      >
        <div style={{ fontSize: 14.5, fontWeight: 700, color: colors.text }}>
          Add this crop to the dataset
        </div>
        <div style={{ fontSize: 11.5, color: colors.textMuted }}>
          Virtual alias of {name} — no image file is created; the {size} render
          happens on the fly at deploy time.
        </div>
        {option(
          "replace",
          "Replace the original",
          "The crop takes its place in the dataset — the original stays behind.",
          true,
        )}
        {option(
          "beside",
          "Place beside",
          "Keeps the original and adds the crop as a duplicate (2 samples).",
          false,
        )}
        <button
          type="button"
          onClick={onBack}
          style={{
            background: "none",
            border: "none",
            color: colors.textMuted,
            fontSize: 11,
            cursor: "pointer",
            alignSelf: "flex-start",
          }}
        >
          ← Back to the crop
        </button>
      </div>
    </div>
  );
}
