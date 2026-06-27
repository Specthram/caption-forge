/**
 * The Libraries tab's near-duplicate comparator.
 *
 * A thin binding of the store's ``compare`` state onto the shared
 * :func:`CompareOverlay` — the same draggable wipe bar, synced zoom/pan and
 * side-by-side toggle the dataset composer uses, so a duplicate is judged
 * the same way wherever it is met.
 */

import { colors } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import { CompareOverlay } from "./CompareOverlay";

export function LookalikeCompare() {
  const compare = useUiStore((state) => state.compare);
  const close = useUiStore((state) => state.closeCompare);

  if (!compare.open) return null;

  return (
    <CompareOverlay
      title={`Near-duplicate — ${compare.leftName} ↔ ${compare.rightName}`}
      subtitle="perceptual hashes matched · pick the one to keep"
      candidate={{
        src: compare.leftSrc,
        name: compare.leftName,
        badge: "best",
        badgeColor: colors.accent,
      }}
      reference={{
        src: compare.rightSrc,
        name: compare.rightName,
        badge: "lookalike",
        badgeColor: colors.textMuted,
      }}
      onClose={close}
      zIndex={60}
    />
  );
}
