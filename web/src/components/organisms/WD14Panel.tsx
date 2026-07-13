/** Collapsible WD14 auto-tagger panel under the Media toolbar. */

import { useEffect, useState } from "react";
import {
  useGroundingEnabled,
  useRunTagger,
  useSettings,
  useTaggerModels,
} from "../../api/hooks";
import { colors, font } from "../../design/tokens";
import { Button, Label, Segmented, Slider } from "../atoms";

/** Read one numeric preference out of the untyped settings payload. */
function savedNumber(value: unknown, fallback: number): number {
  return typeof value === "number" ? value : fallback;
}

export function WD14Panel({
  filterTagIds,
  excludeTagIds,
  match,
  pageKeys,
  focusKey,
}: {
  filterTagIds: number[];
  excludeTagIds: number[];
  match: string;
  pageKeys: string[];
  focusKey: string | null;
}) {
  const models = useTaggerModels();
  const settings = useSettings();
  const groundingEnabled = useGroundingEnabled();
  const run = useRunTagger();

  const [source, setSource] = useState("");
  const [general, setGeneral] = useState(0.35);
  const [character, setCharacter] = useState(0.85);
  const [scope, setScope] = useState("page");
  const [groundAfter, setGroundAfter] = useState(false);
  const [note, setNote] = useState("");

  // Seed the controls from the Settings the Auto-tags index step uses, so a
  // manual run matches what the pipeline would have written — model
  // included (the configured tagger, not the factory default).
  useEffect(() => {
    if (source) return;
    const saved =
      typeof settings.data?.autotag_source === "string"
        ? settings.data.autotag_source
        : "";
    if (saved) setSource(saved);
    else if (models.data) setSource(models.data.default_source);
  }, [models.data, settings.data, source]);
  useEffect(() => {
    const saved = settings.data;
    if (!saved) return;
    setGeneral((value) => savedNumber(saved.autotag_general, value));
    setCharacter((value) => savedNumber(saved.autotag_character, value));
  }, [settings.data]);

  const start = () => {
    if (!source) return;
    const mediaIds =
      scope === "this" && focusKey
        ? [Number(focusKey)]
        : scope === "page"
          ? pageKeys.map(Number)
          : [];
    run.mutate(
      {
        source,
        general,
        character,
        scope: scope === "filtered" ? "filtered" : "media",
        media_ids: mediaIds,
        filter_tag_ids: filterTagIds,
        exclude_tag_ids: excludeTagIds,
        match,
        ground_after: groundAfter,
      },
      {
        onSuccess: (data) =>
          setNote(`Queued WD14 on ${data.count} media — see Jobs.`),
      },
    );
  };

  return (
    <div
      style={{
        padding: "12px 16px",
        borderBottom: `1px solid ${colors.border}`,
        background: colors.card,
        display: "flex",
        gap: 20,
        flexWrap: "wrap",
        alignItems: "flex-end",
      }}
    >
      <div style={{ minWidth: 220 }}>
        <Label>Tagger model</Label>
        <select
          value={source}
          onChange={(event) => setSource(event.target.value)}
          style={selectStyle}
        >
          {models.data?.models.map((model) => (
            <option key={model.source} value={model.source}>
              {model.label}
              {model.available ? " ✓" : ""}
            </option>
          ))}
        </select>
      </div>
      <div style={{ width: 150 }}>
        <Label>General ≥ {general.toFixed(2)}</Label>
        <Slider min={0} max={1} step={0.01} value={general} onChange={setGeneral} />
      </div>
      <div style={{ width: 150 }}>
        <Label>Character ≥ {character.toFixed(2)}</Label>
        <Slider
          min={0}
          max={1}
          step={0.01}
          value={character}
          onChange={setCharacter}
        />
      </div>
      <div>
        <Label>Scope</Label>
        <Segmented
          value={scope}
          onChange={setScope}
          options={[
            { value: "this", label: "This image" },
            { value: "page", label: "Page" },
            { value: "filtered", label: "Filtered" },
          ]}
        />
      </div>
      {groundingEnabled && (
        <label
          title="A WD14 tagger is confident, not correct. SigLIP then checks each tag against the pixels."
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 11.5,
            color: colors.textMuted,
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={groundAfter}
            onChange={(event) => setGroundAfter(event.target.checked)}
          />
          ◈ Ground tags after
        </label>
      )}
      <Button variant="accent" onClick={start} disabled={run.isPending}>
        ✨ Run tagger
      </Button>
      {note && (
        <span style={{ fontSize: 11, color: colors.ok, fontFamily: font.mono }}>
          {note}
        </span>
      )}
    </div>
  );
}

const selectStyle = {
  width: "100%",
  padding: "6px 8px",
  borderRadius: 6,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 12,
} as const;
