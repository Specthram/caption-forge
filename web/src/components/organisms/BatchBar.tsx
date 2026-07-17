/** Floating batch-action bar shown when the selection is non-empty. */

import { useState } from "react";
import {
  useAddTag,
  useDeployMedia,
  useGenerate,
  useProfiles,
  useRemoveFromDataset,
  useTagCategories,
} from "../../api/hooks";
import { colors, font } from "../../design/tokens";
import { useUiStore } from "../../store/uiStore";
import { useSelectionStore } from "../../store/selectionStore";
import { useCaptionStore } from "../../store/captionStore";
import { Button } from "../atoms";

export function BatchBar() {
  const datasetId = useUiStore((state) => state.datasetId);
  const captionType = useUiStore((state) => state.captionType);
  const selected = useSelectionStore((state) => state.selected);
  const clear = useSelectionStore((state) => state.clear);
  const gen = useCaptionStore();

  const generate = useGenerate();
  const profiles = useProfiles();
  const deployMedia = useDeployMedia();
  const addTag = useAddTag();
  const removeFromDataset = useRemoveFromDataset();
  const categories = useTagCategories();

  const [tagInput, setTagInput] = useState("");
  const [tagOpen, setTagOpen] = useState(false);

  if (selected.size === 0 || datasetId == null) return null;
  const keys = Array.from(selected);
  const ids = keys.map(Number);

  const generateSelected = () =>
    generate.mutate({
      dataset_id: datasetId,
      caption_type: captionType,
      media_ids: ids,
      exclude_ids: Array.from(gen.locked).map(Number),
      prompt: gen.prompt,
      profile_id: profiles.data?.active_id ?? null,
      seed: gen.seed ? Number(gen.seed) : null,
      review_after: gen.reviewAfter,
      review_judge_profile_id: gen.reviewAfter
        ? (profiles.data?.judge_id ?? null)
        : null,
      ground_after: gen.groundAfter,
      recaption: gen.recaption,
      unload_after: gen.unloadAfter,
    });

  const addTagToAll = async () => {
    const category = categories.data?.categories[0];
    const name = tagInput.trim();
    if (!name || !category) return;
    for (const key of keys) {
      await addTag.mutateAsync({ key, name, category_id: category.id });
    }
    setTagInput("");
    setTagOpen(false);
  };

  const removeSelected = async () => {
    for (const key of keys) {
      await removeFromDataset.mutateAsync({
        key,
        dataset_id: datasetId,
        caption_type: captionType,
      });
    }
    clear();
  };

  return (
    <div
      style={{
        position: "fixed",
        bottom: 20,
        left: "50%",
        transform: "translateX(-50%)",
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "10px 16px",
        borderRadius: 12,
        background: colors.panel,
        border: `1px solid ${colors.borderHover}`,
        boxShadow: "0 12px 40px rgba(0,0,0,0.5)",
        zIndex: 30,
      }}
    >
      <span
        style={{ fontFamily: font.mono, fontSize: 12, color: colors.accent }}
      >
        {selected.size} selected
      </span>
      <Button variant="accent" onClick={generateSelected}>
        ✦ Generate
      </Button>
      <Button onClick={() => gen.lockMany(keys, true)}>Lock</Button>
      <Button onClick={() => gen.lockMany(keys, false)}>Unlock</Button>
      {tagOpen ? (
        <span style={{ display: "inline-flex", gap: 6 }}>
          <input
            autoFocus
            value={tagInput}
            onChange={(event) => setTagInput(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && addTagToAll()}
            placeholder="tag name"
            style={{
              padding: "5px 8px",
              borderRadius: 6,
              border: `1px solid ${colors.borderControl}`,
              background: colors.input,
              color: colors.text,
              fontSize: 12,
              width: 120,
            }}
          />
          <Button onClick={addTagToAll}>Add</Button>
        </span>
      ) : (
        <Button onClick={() => setTagOpen(true)}>Add tags…</Button>
      )}
      <Button
        onClick={() =>
          deployMedia.mutate({
            dataset_id: datasetId,
            keys,
            caption_type: captionType,
          })
        }
      >
        ⇪ Deploy
      </Button>
      <Button variant="danger" onClick={removeSelected}>
        Remove
      </Button>
      <button
        onClick={clear}
        style={{
          background: "transparent",
          border: "none",
          color: colors.textMuted,
          cursor: "pointer",
          fontSize: 13,
        }}
      >
        ✕ clear
      </button>
    </div>
  );
}
