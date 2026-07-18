/** System workspace: database, runtime, maintenance, SQLite explorer. */

import { useState } from "react";
import {
  useBackupNow,
  useCleanupCategory,
  useCleanupPurge,
  useDbDeleteRow,
  useDbQuery,
  usePurge,
  useRestart,
  useRestoreBackup,
  useSystemDatabase,
  useSystemRuntime,
} from "../api/hooks";
import type {
  CleanupCategory,
  CleanupCount,
  CleanupResult,
  DbQueryResult,
} from "../api/types";
import { colors, font } from "../design/tokens";
import { Button, Dot, Label, Spinner } from "../components/atoms";

function mb(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
  }
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

// Static copy per cleanup category (see the System handoff). The live count
// and reclaimable size come from the API; ``badge`` turns them into the
// right-hand text and ``note`` describes what a purge did.
const dbNote = (result: CleanupResult) => `${result.purged} purged · vacuumed`;
const fileNote = (result: CleanupResult) =>
  `${result.purged} removed · ${mb(result.bytes)} freed`;
const sizedBadge = (unit: string) => (info: CleanupCount) =>
  `${info.count} ${unit}${info.bytes > 0 ? ` · ${mb(info.bytes)}` : ""}`;

const CLEANUP_ROWS: {
  id: CleanupCategory;
  title: string;
  desc: string;
  badge: (info: CleanupCount) => string;
  note: (result: CleanupResult) => string;
}[] = [
  {
    id: "media",
    title: "Orphan media",
    desc:
      "Deletes media rows referenced by no file and no dataset — dead rows " +
      "left behind, invisible in every grid. Their tags and caption history " +
      "cascade away.",
    badge: sizedBadge("media"),
    note: dbNote,
  },
  {
    id: "captions",
    title: "Unused caption versions",
    desc:
      "Drops every caption version that is neither active nor pinned. The " +
      "current caption of each media is kept.",
    badge: sizedBadge("versions"),
    note: dbNote,
  },
  {
    id: "dataset_captions",
    title: "Captions outside datasets",
    desc:
      "Deletes the whole caption history of media that belong to no " +
      "dataset. The media stay in their library; re-adding one later " +
      "starts its captions from scratch.",
    badge: sizedBadge("captions"),
    note: dbNote,
  },
  {
    id: "claims",
    title: "Claims history (grounding)",
    desc:
      "Clears every stored SigLIP caption grounding — the claims and their " +
      "scores. Recomputed on demand the next time you ground a caption.",
    badge: sizedBadge("claims"),
    note: dbNote,
  },
  {
    id: "quality",
    title: "Quality scores",
    desc:
      "Deletes every stored IQA quality score. Recomputed by the " +
      "Libraries → Quality action (heavy models, GPU).",
    badge: sizedBadge("scores"),
    note: dbNote,
  },
  {
    id: "embeddings",
    title: "Embeddings",
    desc:
      "Deletes the DINOv2 / depth embedding vectors behind the diversity " +
      "map and the auto-builder. Recomputed by the Libraries → Embeddings " +
      "action.",
    badge: sizedBadge("vectors"),
    note: dbNote,
  },
  {
    id: "index",
    title: "Index data (dims & hashes)",
    desc:
      "Resets the per-media dimensions, perceptual hashes and image stats. " +
      "Lookalikes and resolution filters go blind until the next " +
      "Libraries → Index run.",
    badge: sizedBadge("media"),
    note: dbNote,
  },
  {
    id: "crops",
    title: "Rendered crops cache",
    desc:
      "Deletes the materialized crop PNGs. Crops are virtual rectangles — " +
      "the pixels are re-rendered on demand at the next read.",
    badge: sizedBadge("files"),
    note: fileNote,
  },
  {
    id: "patches",
    title: "Orphan patches",
    desc:
      "Removes crop / watermark cache files whose source media or dataset no " +
      "longer exists.",
    badge: sizedBadge("files"),
    note: fileNote,
  },
  {
    id: "wm_backups",
    title: "Watermark backups",
    desc:
      "Deletes the pre-patch original backups. Patched media keep their " +
      "current pixels; only the “restore original” option is lost.",
    badge: sizedBadge("files"),
    note: fileNote,
  },
  {
    id: "thumbs",
    title: "Thumbnail cache",
    desc:
      "Deletes every generated thumbnail. They are rebuilt on demand as you " +
      "browse — first load is slower.",
    badge: sizedBadge("files"),
    note: fileNote,
  },
];

export function SystemView() {
  const database = useSystemDatabase();
  const runtime = useSystemRuntime();
  const backup = useBackupNow();
  const restore = useRestoreBackup();
  const purge = usePurge();
  const restart = useRestart();
  const cleanupPurge = useCleanupPurge();
  const runQuery = useDbQuery();
  const deleteRow = useDbDeleteRow();

  const [armed, setArmed] = useState<CleanupCategory | null>(null);
  const [notes, setNotes] = useState<
    Partial<Record<CleanupCategory, string>>
  >({});
  const [dev, setDev] = useState(false);
  const [sql, setSql] = useState("SELECT * FROM media LIMIT 200;");
  const [table, setTable] = useState<string | null>(null);
  const [result, setResult] = useState<DbQueryResult | null>(null);
  const [note, setNote] = useState("");

  const runTable = (name: string) => {
    setTable(name);
    const query = `SELECT * FROM ${name} LIMIT 200;`;
    setSql(query);
    runQuery.mutate(query, { onSuccess: setResult });
  };
  const run = () => {
    setTable(null);
    runQuery.mutate(sql, {
      onSuccess: setResult,
      onError: (error) => setNote(String(error)),
    });
  };
  const idIndex = result ? result.headers.indexOf("id") : -1;

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 24, overflowY: "auto", height: "100%" }}>
      <Card title="Database">
        <Row label="Path">
          <span style={{ fontFamily: font.mono, fontSize: 11 }}>{database.data?.path}</span>
        </Row>
        <Row label="Size">
          <span>{database.data ? mb(database.data.size_bytes) : "…"}</span>
          <Dot color={colors.ok} /> healthy
        </Row>
        <div style={{ display: "flex", gap: 8, margin: "10px 0" }}>
          <Button variant="accent" disabled={backup.isPending} onClick={() => backup.mutate()}>
            {backup.isPending ? <Spinner size={12} /> : "Back up now"}
          </Button>
        </div>
        {(database.data?.backups.length ?? 0) > 0 && (
          <div style={{ marginTop: 6 }}>
            <Label>Backups</Label>
            {database.data?.backups.map((item) => (
              <div key={item.filename} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 11.5, marginBottom: 4 }}>
                <span style={{ fontFamily: font.mono, flex: 1 }}>{item.filename}</span>
                <span style={{ color: colors.textFaint }}>{mb(item.size_bytes)}</span>
                <a
                  onClick={() => {
                    if (window.confirm(`Restore ${item.filename} over the live database?`)) {
                      restore.mutate(item.filename);
                    }
                  }}
                  style={{ cursor: "pointer" }}
                >
                  restore
                </a>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card title="Runtime">
        <Row label="Python">{runtime.data?.python}</Row>
        <Row label="CUDA">{runtime.data?.cuda ?? "—"}</Row>
        <Row label="GPU">{runtime.data?.gpu ?? "—"}</Row>
        <Row label="VRAM">
          {runtime.data?.vram_used_gb ?? "?"} / {runtime.data?.vram_total_gb ?? "?"} GB
        </Row>
        <Row label="Thumbnail cache">
          {runtime.data ? mb(runtime.data.thumbnail_cache_bytes) : "…"}
        </Row>
      </Card>

      <Card title="Maintenance">
        <div style={{ display: "flex", gap: 10 }}>
          <Button disabled={purge.isPending} onClick={() => purge.mutate()}>
            Purge RAM &amp; VRAM
          </Button>
          <Button
            style={{ color: colors.warn, borderColor: colors.warn }}
            disabled={restart.isPending}
            onClick={() => {
              if (window.confirm("Restart the server? Running jobs stop.")) {
                restart.mutate();
              }
            }}
          >
            Restart server
          </Button>
        </div>

        <div
          style={{ borderTop: `1px solid ${colors.border}`, marginTop: 14 }}
        />
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            margin: "12px 0",
          }}
        >
          <Label>Database cleanup</Label>
          <span style={{ flex: 1 }} />
          <span
            style={{ fontSize: 10, fontFamily: font.mono, color: colors.textFaint }}
          >
            orphans only — nothing referenced by a dataset is touched
          </span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {CLEANUP_ROWS.map((row) => {
            return (
              <CleanupRow
                key={row.id}
                row={row}
                note={notes[row.id]}
                armed={armed === row.id}
                purging={cleanupPurge.isPending}
                onArm={() => {
                  setNotes((prev) => ({ ...prev, [row.id]: undefined }));
                  setArmed(row.id);
                }}
                onCancel={() => setArmed(null)}
                onPurge={() =>
                  cleanupPurge.mutate(row.id, {
                    onSuccess: (result) => {
                      setNotes((prev) => ({
                        ...prev,
                        [row.id]: row.note(result),
                      }));
                      setArmed(null);
                    },
                  })
                }
              />
            );
          })}
        </div>
      </Card>

      <Card title="Developer mode">
        <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, cursor: "pointer" }}>
          <input type="checkbox" checked={dev} onChange={(e) => setDev(e.target.checked)} />
          Enable the SQLite explorer (read-only + single-row delete)
        </label>

        {dev && (
          <div style={{ marginTop: 14 }}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
              {Object.entries(database.data?.counts ?? {}).map(([name, count]) => (
                <span
                  key={name}
                  onClick={() => runTable(name)}
                  style={{
                    fontSize: 11,
                    fontFamily: font.mono,
                    padding: "2px 8px",
                    borderRadius: 10,
                    cursor: "pointer",
                    background: table === name ? colors.accentTint : colors.raised,
                    color: table === name ? colors.accent : colors.textMuted,
                    border: `1px solid ${colors.borderControl}`,
                  }}
                >
                  {name} {count}
                </span>
              ))}
            </div>
            <textarea
              value={sql}
              onChange={(e) => setSql(e.target.value)}
              rows={3}
              style={{
                width: "100%",
                padding: 8,
                borderRadius: 6,
                border: `1px solid ${colors.borderControl}`,
                background: colors.input,
                color: colors.ok,
                fontFamily: font.mono,
                fontSize: 12,
              }}
            />
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 8 }}>
              <Button onClick={run}>Run</Button>
              {note && <span style={{ color: colors.danger, fontSize: 11 }}>{note}</span>}
            </div>

            {result && (
              <div style={{ marginTop: 12, overflowX: "auto", border: `1px solid ${colors.border}`, borderRadius: 6 }}>
                <table style={{ borderCollapse: "collapse", fontSize: 10.5, fontFamily: font.mono, width: "100%" }}>
                  <thead>
                    <tr style={{ background: colors.toolbar }}>
                      {table && idIndex >= 0 && <th style={cell}></th>}
                      {result.headers.map((header) => (
                        <th key={header} style={cell}>
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.rows.map((row, index) => (
                      <tr key={index}>
                        {table && idIndex >= 0 && (
                          <td style={cell}>
                            <span
                              onClick={() =>
                                deleteRow.mutate(
                                  { table, row_id: Number(row[idIndex]) },
                                  { onSuccess: () => runTable(table) },
                                )
                              }
                              style={{ cursor: "pointer", color: colors.danger }}
                            >
                              ✕
                            </span>
                          </td>
                        )}
                        {row.map((value, col) => (
                          <td key={col} style={cell}>
                            {value == null ? "" : String(value).slice(0, 80)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <div style={{ fontSize: 10.5, color: colors.textFaint, marginTop: 6 }}>
              {result ? `${result.rows.length} rows shown` : ""} · browsing caps at 200 rows.
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

function CleanupRow({
  row,
  note,
  armed,
  purging,
  onArm,
  onCancel,
  onPurge,
}: {
  row: (typeof CLEANUP_ROWS)[number];
  note: string | undefined;
  armed: boolean;
  purging: boolean;
  onArm: () => void;
  onCancel: () => void;
  onPurge: () => void;
}) {
  // Each row loads its own report: a slow scan (patch orphans, big cache
  // trees) spins alone instead of stalling every category.
  const info = useCleanupCategory(row.id);
  const count = info.data?.count ?? 0;
  const empty = count === 0;
  const done = Boolean(note);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 12, fontWeight: 600 }}>{row.title}</span>
          <span
            style={{
              fontSize: 10,
              fontFamily: font.mono,
              padding: "1px 7px",
              border: `1px solid ${colors.borderControl}`,
              borderRadius: 5,
              background: colors.card,
              color: empty ? colors.ok : colors.warn,
              display: "inline-flex",
              alignItems: "center",
              gap: 5,
            }}
          >
            {info.data === undefined ? (
              <>
                <Spinner size={9} /> measuring…
              </>
            ) : empty ? (
              done ? (
                "clean ✓"
              ) : (
                "0"
              )
            ) : (
              row.badge(info.data)
            )}
          </span>
        </div>
        <div
          style={{
            fontSize: 11,
            color: colors.textMuted,
            lineHeight: 1.45,
            marginTop: 2,
          }}
        >
          {row.desc}
        </div>
      </div>
      {note && (
        <span
          style={{ fontSize: 10.5, fontFamily: font.mono, color: colors.ok }}
        >
          {note}
        </span>
      )}
      {armed ? (
        <>
          <span
            onClick={onCancel}
            style={{ fontSize: 11, color: colors.textMuted, cursor: "pointer" }}
          >
            cancel
          </span>
          <Button
            variant="danger"
            style={{ fontWeight: 700 }}
            disabled={purging}
            onClick={onPurge}
          >
            Confirm — delete {count}
          </Button>
        </>
      ) : (
        <Button variant="danger" disabled={empty} onClick={onArm}>
          ⌫ Purge
        </Button>
      )}
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ background: colors.card, border: `1px solid ${colors.border}`, borderRadius: 9, padding: 18, marginBottom: 16 }}>
      <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 12 }}>{title}</div>
      {children}
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, marginBottom: 6 }}>
      <span style={{ width: 130, color: colors.textMuted }}>{label}</span>
      {children}
    </div>
  );
}

const cell = {
  border: `1px solid ${colors.border}`,
  padding: "3px 6px",
  textAlign: "left",
  whiteSpace: "nowrap",
} as const;
