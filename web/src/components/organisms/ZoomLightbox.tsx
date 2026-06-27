/**
 * Full-screen zoom lightbox. Wheel zooms toward the cursor (factor 1.18,
 * clamp 0.4×–14×) with the transform ``translate(tx,ty) scale(s)`` about the
 * viewport centre; drag pans, and a drag over 3px suppresses click-to-close.
 */

import { useRef } from "react";
import { colors, font } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";

export function ZoomLightbox() {
  const zoom = useUiStore((state) => state.zoom);
  const setZoom = useUiStore((state) => state.setZoom);
  const close = useUiStore((state) => state.closeZoom);
  const drag = useRef<{ x: number; y: number; moved: number } | null>(null);

  if (!zoom.open || !zoom.src) return null;

  const onWheel = (event: React.WheelEvent) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18;
    const next = Math.min(14, Math.max(0.4, zoom.scale * factor));
    const rect = event.currentTarget.getBoundingClientRect();
    const cx = event.clientX - rect.left - rect.width / 2;
    const cy = event.clientY - rect.top - rect.height / 2;
    const ratio = next / zoom.scale;
    setZoom({
      scale: next,
      tx: cx - ratio * (cx - zoom.tx),
      ty: cy - ratio * (cy - zoom.ty),
    });
  };

  const onDown = (event: React.MouseEvent) => {
    drag.current = { x: event.clientX, y: event.clientY, moved: 0 };
  };
  const onMove = (event: React.MouseEvent) => {
    if (!drag.current) return;
    const dx = event.clientX - drag.current.x;
    const dy = event.clientY - drag.current.y;
    drag.current.moved += Math.abs(dx) + Math.abs(dy);
    drag.current.x = event.clientX;
    drag.current.y = event.clientY;
    setZoom({ tx: zoom.tx + dx, ty: zoom.ty + dy });
  };
  const onUp = (event: React.MouseEvent) => {
    const moved = drag.current?.moved ?? 0;
    drag.current = null;
    if (moved <= 3 && event.target === event.currentTarget) close();
  };

  return (
    <div
      onWheel={onWheel}
      onMouseDown={onDown}
      onMouseMove={onMove}
      onMouseUp={onUp}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(8,9,11,0.9)",
        zIndex: 60,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
        cursor: "grab",
      }}
    >
      <img
        src={zoom.src}
        alt={zoom.name}
        draggable={false}
        style={{
          maxWidth: "72vmin",
          maxHeight: "72vmin",
          transform: `translate(${zoom.tx}px, ${zoom.ty}px) scale(${zoom.scale})`,
          transformOrigin: "center",
          userSelect: "none",
        }}
      />
      <div
        style={{
          position: "fixed",
          top: 16,
          left: "50%",
          transform: "translateX(-50%)",
          display: "flex",
          gap: 12,
          alignItems: "center",
          fontFamily: font.mono,
          fontSize: 11,
          color: colors.textSecondary,
          background: "rgba(0,0,0,0.5)",
          padding: "6px 12px",
          borderRadius: 20,
        }}
      >
        <span>{zoom.name}</span>
        <span>{Math.round(zoom.scale * 100)}%</span>
        <span style={{ color: colors.textFaint }}>scroll to zoom · Esc</span>
      </div>
    </div>
  );
}
