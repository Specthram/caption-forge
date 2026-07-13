/**
 * Subfolder-mapping wizard (Libraries view).
 *
 * A two-step flow for adding a large source folder without everything landing
 * mixed-up in the gallery. Step 1 picks the source folder (server-side
 * browse); step 2 maps each sub-folder: keep it in the parent, promote it to
 * its own sub-library (excluded from the parent), or skip it — with auto /
 * inherited / manual tag rules that the scan applies and future scans keep.
 *
 * The effective-tags and routing logic mirrors `src/folder_rules.py`; the
 * backend is authoritative (it re-resolves on scan), this only previews.
 */

import { useMemo, useState } from "react";
import {
  useBrowseFolder,
  useCreateLibrary,
  useExistingTagNames,
  useFolderRules,
  useFolderTree,
  usePutFolderRules,
  useTagSearch,
} from "../../api/hooks";
import type {
  FolderRuleInput,
  FolderTreeNode,
  LibrarySource,
} from "../../api/types";
import { colors, font, radii, shadow } from "../../design/tokens";
import { Button } from "../atoms";

type Mode = "keep" | "sublib" | "exclude";
type AutoLevel = "0" | "1" | "all";

interface Rule {
  mode: Mode;
  tags: string[];
  removed: string[];
  subName: string;
}

type Rules = Record<string, Rule>;

const slug = (name: string) => name.trim().replace(/\s+/g, "_");

const DEFAULT_RULE: Rule = {
  mode: "keep",
  tags: [],
  removed: [],
  subName: "",
};

function ruleOf(rules: Rules, node: FolderTreeNode): Rule {
  return rules[node.rel_path] ?? { ...DEFAULT_RULE, subName: node.name };
}

/** One flattened, resolved tree row (order = pre-order DFS). */
interface Row {
  node: FolderTreeNode;
  depth: number;
  owner: string;
  inherited: string[];
  auto: string[];
  manual: string[];
  mode: Mode;
  subName: string;
  /** The raw stored rule (defaults included), for correct edit merges. */
  rule: Rule;
  visible: boolean;
}

interface Resolved {
  rows: Row[];
  libs: { name: string; count: number; sub: boolean; parent: string | null }[];
  excludedFiles: number;
  taggedFiles: number;
  ruleCount: number;
  /** The root folder's own tags (rel_path ""), cascaded to every child. */
  rootTags: string[];
}

/** Walk the tree once, computing rows, per-library counts and tag totals. */
function resolve(
  tree: { own: number; children: FolderTreeNode[] } | undefined,
  rules: Rules,
  autoLevel: AutoLevel,
  parentName: string,
  expanded: Set<string>,
): Resolved {
  const rows: Row[] = [];
  const libs: Resolved["libs"] = [
    { name: parentName, count: tree?.own ?? 0, sub: false, parent: null },
  ];
  let excludedFiles = 0;
  let taggedFiles = 0;
  let ruleCount = 0;

  // Root ("") tags apply to every file and seed each child's inherited chain.
  const rootTags = (rules[""] ?? DEFAULT_RULE).tags;
  if (rootTags.length) {
    taggedFiles += tree?.own ?? 0;
    ruleCount += 1;
  }

  const visit = (
    nodes: FolderTreeNode[],
    depth: number,
    owner: string,
    parentVisible: boolean,
    inherited: string[],
  ) => {
    for (const node of nodes) {
      const rule = ruleOf(rules, node);
      const isExc = rule.mode === "exclude";
      const isSub = rule.mode === "sublib";
      const removed = rule.removed;
      const autoOn = autoLevel === "all" || (autoLevel === "1" && depth === 0);
      const s = slug(node.name);
      const auto = autoOn && !removed.includes(s) ? [s] : [];
      const manual = rule.tags.filter(
        (t) => !auto.includes(t) && !removed.includes(t),
      );
      const inhHere = inherited.filter(
        (t) => !removed.includes(t) && !auto.includes(t) && !manual.includes(t),
      );

      let owner2 = owner;
      if (isExc) {
        excludedFiles += node.total;
      } else {
        if (isSub) {
          owner2 = rule.subName || node.name;
          libs.push({ name: owner2, count: 0, sub: true, parent: owner });
        }
        const lib = libs.find((l) => l.name === owner2) ?? libs[0];
        lib.count += node.own;
        if (auto.length + manual.length + inhHere.length) {
          taggedFiles += node.own;
        }
        if (auto.length + manual.length) ruleCount += 1;
      }

      rows.push({
        node,
        depth,
        owner,
        inherited: inhHere,
        auto,
        manual,
        mode: rule.mode,
        subName: rule.subName || node.name,
        rule,
        visible: parentVisible,
      });

      if (!isExc) {
        visit(
          node.children,
          depth + 1,
          owner2,
          parentVisible && expanded.has(node.rel_path),
          inhHere.concat(auto, manual),
        );
      }
    }
  };

  visit(tree?.children ?? [], 0, parentName, true, rootTags);
  return { rows, libs, excludedFiles, taggedFiles, ruleCount, rootTags };
}

export function SubfolderMappingModal({
  open,
  libraryId = null,
  initialPath = "",
  initialName = "",
  existing,
  onHide,
  onDiscard,
  onApplied,
}: {
  /** When false the modal stays mounted (draft preserved) but renders nothing. */
  open: boolean;
  libraryId?: number | null;
  initialPath?: string;
  initialName?: string;
  /** Every library, so a re-edit can name sub-libraries from their rows. */
  existing?: LibrarySource[];
  /** Backdrop click: hide without losing the draft (reopen restores it). */
  onHide: () => void;
  /** ✕ / Cancel: close and discard the draft. */
  onDiscard: () => void;
  onApplied?: (jobId: string) => void;
}) {
  // Re-editing an existing library jumps straight to the mapping step.
  const [step, setStep] = useState<"pick" | "map">(
    libraryId != null ? "map" : "pick",
  );
  const [pickPath, setPickPath] = useState(initialPath);
  const [root, setRoot] = useState(libraryId != null ? initialPath : "");
  const [name, setName] = useState(initialName);
  const [autoLevel, setAutoLevel] = useState<AutoLevel>("0");
  const [rules, setRules] = useState<Rules>({});
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [tagEdit, setTagEdit] = useState<string | null>(null);
  const [bulkTag, setBulkTag] = useState("");
  const [seeded, setSeeded] = useState(false);

  const createLibrary = useCreateLibrary();
  const putRules = usePutFolderRules();
  const tree = useFolderTree(root, step === "map" && !!root);
  const saved = useFolderRules(libraryId, step === "map");

  // Seed the rules/level from an existing mapping the first time it loads.
  if (
    !seeded &&
    saved.data &&
    libraryId != null &&
    step === "map" &&
    !tree.isFetching
  ) {
    const next: Rules = {};
    const nameById = new Map(
      (existing ?? []).map((lib) => [lib.id, lib.name]),
    );
    for (const rule of saved.data.rules) {
      next[rule.rel_path] = {
        mode: rule.mode,
        tags: rule.tags,
        removed: rule.removed,
        subName:
          (rule.sub_library_id != null &&
            nameById.get(rule.sub_library_id)) ||
          rule.rel_path.split("/").pop() ||
          "",
      };
    }
    setRules(next);
    setAutoLevel((saved.data.auto_tag_level as AutoLevel) || "0");
    setSeeded(true);
  }

  const parentName =
    name.trim() ||
    (existing?.find((l) => l.id === libraryId)?.name ?? "") ||
    root.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ||
    "library";

  const res = useMemo(
    () => resolve(tree.data, rules, autoLevel, parentName, expanded),
    [tree.data, rules, autoLevel, parentName, expanded],
  );

  // Auto tags (folder names) are coloured green when they match an existing
  // tag and amber when applying will create them — checked in one batch.
  const autoNames = useMemo(() => {
    const set = new Set<string>();
    for (const row of res.rows) row.auto.forEach((tag) => set.add(tag));
    return [...set];
  }, [res]);
  const existingQuery = useExistingTagNames(autoNames, step === "map");
  const existingTags = useMemo(
    () => new Set(existingQuery.data?.existing ?? []),
    [existingQuery.data],
  );

  // Every tag typed anywhere in this draft (manual + root), so the inline
  // tag search can re-offer a just-created tag before it is persisted.
  const draftTags = useMemo(() => {
    const set = new Set<string>();
    for (const rule of Object.values(rules)) {
      rule.tags.forEach((tag) => set.add(tag));
    }
    return [...set];
  }, [rules]);

  const patch = (relPath: string, node: FolderTreeNode, change: Partial<Rule>) =>
    setRules((prev) => ({
      ...prev,
      [relPath]: { ...ruleOf(prev, node), ...change },
    }));

  const setAllModes = (mode: Mode) =>
    setRules((prev) => {
      const next: Rules = { ...prev };
      const walk = (nodes: FolderTreeNode[]) => {
        for (const node of nodes) {
          next[node.rel_path] = { ...ruleOf(next, node), mode };
          walk(node.children);
        }
      };
      walk(tree.data?.children ?? []);
      return next;
    });

  const bulkPatch = (change: Partial<Rule>, addTag?: string) =>
    setRules((prev) => {
      const next: Rules = { ...prev };
      const index = new Map<string, FolderTreeNode>();
      const walk = (nodes: FolderTreeNode[]) => {
        for (const node of nodes) {
          index.set(node.rel_path, node);
          walk(node.children);
        }
      };
      walk(tree.data?.children ?? []);
      for (const relPath of selected) {
        const node = index.get(relPath);
        if (!node) continue;
        const base = { ...ruleOf(next, node), ...change };
        if (addTag && !base.tags.includes(addTag)) {
          base.tags = base.tags.concat([addTag]);
          base.removed = base.removed.filter((t) => t !== addTag);
        }
        next[relPath] = base;
      }
      return next;
    });

  const toggleExpand = (relPath: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(relPath)) next.delete(relPath);
      else next.add(relPath);
      return next;
    });

  const toggleSelect = (relPath: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(relPath)) next.delete(relPath);
      else next.add(relPath);
      return next;
    });

  const buildPayload = (): FolderRuleInput[] =>
    Object.entries(rules)
      .filter(
        ([, rule]) =>
          rule.mode !== "keep" || rule.tags.length || rule.removed.length,
      )
      .map(([relPath, rule]) => ({
        rel_path: relPath,
        mode: rule.mode,
        tags: rule.tags,
        removed: rule.removed,
        sub_name: rule.mode === "sublib" ? rule.subName : null,
      }));

  const apply = () => {
    const mapping = {
      auto_tag_level: autoLevel,
      rules: buildPayload(),
      name: parentName,
    };
    const put = (id: number) =>
      putRules.mutate(
        { libraryId: id, mapping },
        { onSuccess: (data) => onApplied?.(data.job_id) },
      );
    if (libraryId != null) {
      put(libraryId);
    } else {
      createLibrary.mutate(
        { name: parentName, path: root, recursive: true },
        { onSuccess: (data) => put(data.id) },
      );
    }
  };

  const applying = createLibrary.isPending || putRules.isPending;

  // Hidden (backdrop-dismissed): stay mounted so the draft survives; a reopen
  // re-shows this same instance with all its state intact.
  if (!open) return null;

  return (
    <div style={backdrop} onClick={onHide}>
      <div
        style={step === "pick" ? pickPanel : mapPanel}
        onClick={(event) => event.stopPropagation()}
      >
        {step === "pick" ? (
          <PickStep
            pickPath={pickPath}
            setPickPath={setPickPath}
            onClose={onDiscard}
            onConfirm={() => {
              setRoot(pickPath);
              setStep("map");
            }}
          />
        ) : (
          <MapStep
            root={root}
            rootNode={{
              rel_path: "",
              name: parentName,
              own: tree.data?.own ?? 0,
              total: tree.data?.total ?? 0,
              samples: [],
              children: [],
            }}
            tree={tree.data}
            loading={tree.isLoading}
            res={res}
            autoLevel={autoLevel}
            setAutoLevel={setAutoLevel}
            selected={selected}
            expanded={expanded}
            toggleExpand={toggleExpand}
            toggleSelect={toggleSelect}
            clearSelection={() => setSelected(new Set())}
            setAllModes={setAllModes}
            bulkPatch={bulkPatch}
            bulkTag={bulkTag}
            setBulkTag={setBulkTag}
            patch={patch}
            tagEdit={tagEdit}
            setTagEdit={setTagEdit}
            existing={existingTags}
            draftTags={draftTags}
            parentName={parentName}
            name={name}
            setName={setName}
            nameEditable={true}
            canBack={libraryId == null}
            onBack={() => setStep("pick")}
            onClose={onDiscard}
            onApply={apply}
            applying={applying}
          />
        )}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- step 1 */

function PickStep({
  pickPath,
  setPickPath,
  onClose,
  onConfirm,
}: {
  pickPath: string;
  setPickPath: (path: string) => void;
  onClose: () => void;
  onConfirm: () => void;
}) {
  const listing = useBrowseFolder(pickPath, true);
  const data = listing.data;
  const canConfirm = !!data && !data.is_root && !!data.path;

  const [filter, setFilter] = useState("");
  const query = filter.trim().toLowerCase();
  // Clear the quick filter whenever we descend/ascend to a new folder.
  const go = (path: string) => {
    setFilter("");
    setPickPath(path);
  };
  const entries = (data?.entries ?? []).filter(
    (entry) => !query || entry.name.toLowerCase().includes(query),
  );

  return (
    <>
      <ModalHeader
        title="Add folder"
        subtitle="step 1 / 2 — choose the source folder"
        onClose={onClose}
      />
      <div style={pickToolbar}>
        {["C:\\", "D:\\"].map((drive) => {
          const on = pickPath.toUpperCase().startsWith(drive.toUpperCase());
          return (
            <span
              key={drive}
              onClick={() => go(drive)}
              style={{
                ...drivePill,
                color: on ? colors.accent : colors.textMutedAlt,
                background: on ? colors.accentTint : "transparent",
                border: `1px solid ${on ? colors.accentBorder : colors.borderControl}`,
              }}
            >
              {drive.replace("\\", "/")}
            </span>
          );
        })}
        <span style={{ color: colors.textFaint, margin: "0 4px" }}>·</span>
        <span style={crumb}>{data?.is_root ? "This PC" : data?.path || "…"}</span>
      </div>
      {data && !data.is_root && (
        <div style={pickSearchWrap}>
          <div style={searchBar}>
            <span style={{ color: colors.textFaint, fontSize: 12 }}>⌕</span>
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter folders here…"
              style={searchInput}
            />
            {query && (
              <span style={{ ...crumb, fontSize: 10 }}>
                {entries.length} match{entries.length === 1 ? "" : "es"}
              </span>
            )}
            {filter && (
              <span style={linkBtn} onClick={() => setFilter("")}>
                clear
              </span>
            )}
          </div>
        </div>
      )}
      <div style={pickBody}>
        {data && data.parent !== null && !query && (
          <PickRow label=".." glyph="⬆" onClick={() => go(data.parent ?? "")} />
        )}
        {entries.map((entry) => (
          <PickRow
            key={entry.path}
            label={entry.name}
            glyph={data?.is_root ? "🖴" : "▸"}
            chevron={!data?.is_root}
            onClick={() => go(entry.path)}
          />
        ))}
        {data && !data.is_root && data.entries.length === 0 && (
          <div style={{ padding: 14, color: colors.textFaint, fontSize: 12 }}>
            No sub-folders here.
          </div>
        )}
        {data && !data.is_root && data.entries.length > 0 && query && entries.length === 0 && (
          <div style={{ padding: 14, color: colors.textFaint, fontSize: 12 }}>
            No folder matches “{filter.trim()}”.
          </div>
        )}
      </div>
      <div style={footer}>
        <span style={{ ...crumb, flex: 1 }}>
          {canConfirm
            ? `selected: ${data?.path}`
            : "Open a drive, then a folder."}
        </span>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="accent" disabled={!canConfirm} onClick={onConfirm}>
          Use this folder → map subfolders
        </Button>
      </div>
    </>
  );
}

function PickRow({
  label,
  glyph,
  chevron = false,
  onClick,
}: {
  label: string;
  glyph: string;
  chevron?: boolean;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      style={pickRow}
      onMouseEnter={(e) => (e.currentTarget.style.background = colors.raised)}
      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
    >
      <span style={{ color: colors.warn, fontSize: 12 }}>{glyph}</span>
      <span style={{ flex: 1, fontSize: 12.5, fontWeight: 600 }}>{label}</span>
      {chevron && <span style={{ color: colors.textFaint }}>›</span>}
    </div>
  );
}

/* ---------------------------------------------------------------- step 2 */

interface MapStepProps {
  root: string;
  /** Synthetic node for the pinned, taggable root row (rel_path ""). */
  rootNode: FolderTreeNode;
  tree: { total: number; children: FolderTreeNode[] } | undefined;
  loading: boolean;
  res: Resolved;
  autoLevel: AutoLevel;
  setAutoLevel: (level: AutoLevel) => void;
  selected: Set<string>;
  expanded: Set<string>;
  toggleExpand: (relPath: string) => void;
  toggleSelect: (relPath: string) => void;
  clearSelection: () => void;
  setAllModes: (mode: Mode) => void;
  bulkPatch: (change: Partial<Rule>, addTag?: string) => void;
  bulkTag: string;
  setBulkTag: (value: string) => void;
  patch: (relPath: string, node: FolderTreeNode, change: Partial<Rule>) => void;
  tagEdit: string | null;
  setTagEdit: (relPath: string | null) => void;
  /** Auto-tag names that already exist as tags (green vs amber chips). */
  existing: Set<string>;
  /** Tags created anywhere in this draft, for the inline tag search. */
  draftTags: string[];
  parentName: string;
  /** The user-entered library name (empty = fall back to the folder name). */
  name: string;
  setName: (value: string) => void;
  /** Only a newly-created library can be renamed here. */
  nameEditable: boolean;
  canBack: boolean;
  onBack: () => void;
  onClose: () => void;
  onApply: () => void;
  applying: boolean;
}

const AUTO_HINT: Record<AutoLevel, string> = {
  "0": "no automatic name tags — add tags manually",
  "1": "each top-level subfolder gets its name as a tag — children inherit it",
  all: "every folder gets its own name as a tag — children also inherit parent tags",
};

function MapStep(props: MapStepProps) {
  const { res, tree } = props;
  const folderCount = res.rows.length;
  const shown = res.rows.filter((row) => row.visible);

  return (
    <>
      <div style={mapHeader}>
        {props.canBack && (
          <span style={backBtn} onClick={props.onBack}>
            ‹ folder
          </span>
        )}
        <span style={{ fontSize: 14, fontWeight: 600 }}>Subfolder mapping</span>
        <span style={{ ...crumb, flex: 1 }}>
          {props.root} · {folderCount} subfolders · {tree?.total ?? 0} files
        </span>
        <span style={closeX} onClick={props.onClose}>
          ✕
        </span>
      </div>

      <div style={explainer}>
        Tag each subfolder, promote it to its own{" "}
        <span style={{ color: colors.accent, fontWeight: 600 }}>
          sub-library
        </span>{" "}
        (excluded from the parent), or skip it entirely — so nothing lands
        mixed-up in the gallery.
      </div>

      <div style={autoToolbar}>
        <span style={{ fontSize: 11.5, fontWeight: 600 }}>
          Auto-tag folder names
        </span>
        <div style={segGroup}>
          {(
            [
              ["0", "off"],
              ["1", "top level"],
              ["all", "all levels"],
            ] as [AutoLevel, string][]
          ).map(([value, label]) => {
            const on = props.autoLevel === value;
            return (
              <span
                key={value}
                onClick={() => props.setAutoLevel(value)}
                style={{
                  ...segItem,
                  color: on ? colors.greenAlt : colors.textMuted,
                  background: on ? "#16211a" : "transparent",
                }}
              >
                {label}
              </span>
            );
          })}
        </div>
        <span style={{ ...crumb, flex: 1 }}>{AUTO_HINT[props.autoLevel]}</span>
        <span style={legendChip("#e6e7ea", "#2c2f38", "solid", "#3a3d47")}>
          #manual
        </span>
        <span style={legendChip("#8fc796", "#16211a", "solid", "#2a4030")}>
          #auto
        </span>
        <span style={legendChip("#8b8e98", "transparent", "dashed", "#3a3d47")}>
          ⤷ #inherited
        </span>
      </div>

      {props.selected.size > 0 && (
        <BulkBar {...props} />
      )}

      <div style={mapBody}>
        <div style={treePane}>
          <RootRow {...props} />
          <div style={treeHeader}>
            <span style={crumb}>{folderCount} folders</span>
            <div style={{ flex: 1 }} />
            <span
              style={{ fontSize: 10.5, fontWeight: 600, color: colors.textMuted }}
            >
              All folders ↓
            </span>
            <div style={modeGroupBox}>
              <AllModeBtn
                label="✓ keep"
                color={colors.textSecondary}
                tint="#2c2f38"
                onClick={() => props.setAllModes("keep")}
              />
              <AllModeBtn
                label="◫ library"
                color={colors.accent}
                tint={colors.accentTint}
                onClick={() => props.setAllModes("sublib")}
              />
              <AllModeBtn
                label="⊘ skip"
                color={colors.danger}
                tint="#2a1715"
                onClick={() => props.setAllModes("exclude")}
              />
            </div>
          </div>
          {props.loading && (
            <div style={{ padding: 20, color: colors.textMuted }}>
              Reading folders…
            </div>
          )}
          {shown.map((row) => (
            <TreeRow key={row.node.rel_path} row={row} {...props} />
          ))}
          {!props.loading && res.rows.length === 0 && (
            <div style={{ padding: 20, color: colors.textFaint, fontSize: 12 }}>
              This folder has no sub-folders — apply to add it as one library.
            </div>
          )}
        </div>

        <SummaryRail
          libs={res.libs}
          excludedFiles={res.excludedFiles}
          taggedFiles={res.taggedFiles}
          ruleCount={res.ruleCount}
        />
      </div>

      <div style={footer}>
        <span style={{ ...crumb, flex: 1 }}>
          reopen anytime — library page → ⊞ Subfolder mapping
        </span>
        <Button onClick={props.onClose}>Cancel</Button>
        <Button variant="accent" loading={props.applying} onClick={props.onApply}>
          ▶ Apply mapping &amp; scan
        </Button>
      </div>
    </>
  );
}

function AllModeBtn({
  label,
  color,
  tint,
  onClick,
}: {
  label: string;
  color: string;
  tint: string;
  onClick: () => void;
}) {
  return (
    <span
      onClick={onClick}
      style={{
        padding: "3px 8px",
        fontFamily: font.mono,
        fontSize: 10,
        color,
        cursor: "pointer",
        borderRadius: 5,
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = tint)}
      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
    >
      {label}
    </span>
  );
}

function BulkBar(props: MapStepProps) {
  return (
    <div style={bulkBar}>
      <span style={{ color: colors.accent, fontFamily: font.mono, fontSize: 11 }}>
        {props.selected.size} selected
      </span>
      <input
        value={props.bulkTag}
        onChange={(e) => props.setBulkTag(e.target.value)}
        placeholder="tag"
        style={{ ...smallInput, width: 120 }}
      />
      <Button
        onClick={() => {
          const t = props.bulkTag.trim();
          if (t) props.bulkPatch({}, t);
          props.setBulkTag("");
        }}
      >
        # Add tag to all
      </Button>
      <span style={{ color: colors.textFaint }}>·</span>
      <BulkModeBtn label="✓ keep" onClick={() => props.bulkPatch({ mode: "keep" })} />
      <BulkModeBtn
        label="◫ sub-library"
        color={colors.accent}
        onClick={() => props.bulkPatch({ mode: "sublib" })}
      />
      <BulkModeBtn
        label="⊘ skip"
        color={colors.danger}
        onClick={() => props.bulkPatch({ mode: "exclude" })}
      />
      <div style={{ flex: 1 }} />
      <span style={linkBtn} onClick={props.clearSelection}>
        clear selection
      </span>
    </div>
  );
}

function BulkModeBtn({
  label,
  color = colors.textSecondary,
  onClick,
}: {
  label: string;
  color?: string;
  onClick: () => void;
}) {
  return (
    <span
      onClick={onClick}
      style={{
        fontFamily: font.mono,
        fontSize: 10.5,
        color,
        cursor: "pointer",
        padding: "2px 6px",
      }}
    >
      {label}
    </span>
  );
}

/** Pinned, taggable root row: its tags apply to every file and cascade. */
function RootRow({
  root,
  rootNode,
  res,
  patch,
  tagEdit,
  setTagEdit,
  draftTags,
  name,
  setName,
  nameEditable,
}: MapStepProps) {
  const editing = tagEdit === "";
  const tags = res.rootTags;
  return (
    <div style={rootRowBox}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ color: colors.accent, fontSize: 13 }}>◈</span>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            {nameEditable ? (
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={rootNode.name}
                title="Library name"
                style={{
                  ...smallInput,
                  width: 200,
                  fontSize: 12,
                  fontWeight: 700,
                }}
              />
            ) : (
              <span style={{ fontSize: 12, fontWeight: 700 }}>
                {rootNode.name}
              </span>
            )}
            <span style={rootBadge}>root</span>
          </div>
          <div style={{ ...crumb, fontSize: 9.5 }}>{root} · all files</div>
        </div>
        <span style={{ ...crumb, fontSize: 9.5 }}>
          tags every file — subfolders inherit
        </span>
      </div>
      <div style={chipsZone}>
        {tags.map((tag) => (
          <Chip
            key={`r-${tag}`}
            label={tag}
            color="#e6e7ea"
            bg="#2c2f38"
            borderColor="#3a3d47"
            onRemove={() =>
              patch(rootNode.rel_path, rootNode, {
                tags: tags.filter((t) => t !== tag),
              })
            }
          />
        ))}
        {editing ? (
          <FolderTagInput
            current={tags}
            localTags={draftTags}
            onAdd={(tagName) =>
              patch(rootNode.rel_path, rootNode, {
                tags: dedupe([...tags, tagName]),
              })
            }
            onClose={() => setTagEdit(null)}
          />
        ) : (
          <span style={addChip} onClick={() => setTagEdit("")}>
            + tag
          </span>
        )}
      </div>
    </div>
  );
}

function TreeRow({
  row,
  patch,
  toggleExpand,
  toggleSelect,
  selected,
  expanded,
  tagEdit,
  setTagEdit,
  existing,
  draftTags,
}: { row: Row } & MapStepProps) {
  const { node } = row;
  const isExc = row.mode === "exclude";
  const isSub = row.mode === "sublib";
  const isSel = selected.has(node.rel_path);
  const hasChildren = node.children.length > 0;
  const editing = tagEdit === node.rel_path;

  const modeControl = (
    [
      ["keep", "✓ keep", colors.text, "#2c2f38"],
      ["sublib", "◫ library", colors.accent, colors.accentTint],
      ["exclude", "⊘ skip", colors.danger, "#2a1715"],
    ] as [Mode, string, string, string][]
  ).map(([value, label, activeColor, activeBg]) => {
    const on = row.mode === value;
    return (
      <span
        key={value}
        onClick={() => patch(node.rel_path, node, { mode: value })}
        style={{
          padding: "3px 7px",
          fontFamily: font.mono,
          fontSize: 10,
          borderRadius: 5,
          cursor: "pointer",
          color: on ? activeColor : colors.textMuted,
          background: on ? activeBg : "transparent",
        }}
      >
        {label}
      </span>
    );
  });

  return (
    <div
      style={{
        ...rowBox,
        marginLeft: row.depth * 26,
        opacity: isExc ? 0.55 : 1,
        border: `1px solid ${isSel ? colors.accentBorder : "#1e2026"}`,
        background: isSel ? "#1d1a14" : "#131418",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          onClick={() => hasChildren && !isExc && toggleExpand(node.rel_path)}
          style={{
            width: 12,
            cursor: hasChildren && !isExc ? "pointer" : "default",
            color: colors.textMuted,
            fontSize: 11,
          }}
        >
          {hasChildren && !isExc ? (expanded.has(node.rel_path) ? "▾" : "▸") : ""}
        </span>
        <input
          type="checkbox"
          checked={isSel}
          onChange={() => toggleSelect(node.rel_path)}
        />
        <div style={{ display: "flex", gap: 2 }}>
          {[0, 1, 2].map((i) => (
            <div key={i} style={thumbMini} />
          ))}
        </div>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div
            style={{
              fontSize: 12,
              fontWeight: 600,
              color: isExc
                ? colors.textMuted
                : isSub
                  ? colors.accent
                  : colors.text,
            }}
          >
            {node.name}
          </div>
          <div style={{ ...crumb, fontSize: 9.5 }}>
            {node.total} files
            {hasChildren ? ` · ${node.children.length} subfolders` : ""}
          </div>
        </div>
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 9.5,
            color: isExc
              ? colors.danger
              : isSub
                ? colors.accent
                : colors.textFaint,
          }}
        >
          {isExc ? "⊘ skipped" : isSub ? "◫ own library" : `→ ${row.owner}`}
        </span>
        <div style={modeGroupBox}>{modeControl}</div>
      </div>

      {!isExc && (
        <div style={chipsZone}>
          {row.inherited.map((tag) => (
            <Chip
              key={`i-${tag}`}
              label={`⤷ ${tag}`}
              color="#8b8e98"
              bg="transparent"
              borderStyle="dashed"
              borderColor="#3a3d47"
              onRemove={() =>
                patch(node.rel_path, node, {
                  removed: dedupe([...row.rule.removed, tag]),
                })
              }
            />
          ))}
          {row.auto.map((tag) => {
            const known = existing.has(tag);
            return (
              <Chip
                key={`a-${tag}`}
                label={tag}
                color={known ? "#8fc796" : "#e0b356"}
                bg={known ? "#16211a" : "#211c10"}
                borderColor={known ? "#2a4030" : "#4a3a22"}
                title={
                  known
                    ? "auto — folder name (existing tag)"
                    : "auto — folder name · will be created on apply"
                }
                onRemove={() =>
                  patch(node.rel_path, node, {
                    removed: dedupe([...row.rule.removed, tag]),
                  })
                }
              />
            );
          })}
          {row.manual.map((tag) => (
            <Chip
              key={`m-${tag}`}
              label={tag}
              color="#e6e7ea"
              bg="#2c2f38"
              borderColor="#3a3d47"
              onRemove={() =>
                patch(node.rel_path, node, {
                  tags: row.rule.tags.filter((t) => t !== tag),
                })
              }
            />
          ))}
          {editing ? (
            <FolderTagInput
              current={row.rule.tags}
              localTags={draftTags}
              onAdd={(tagName) =>
                patch(node.rel_path, node, {
                  tags: dedupe([...row.rule.tags, tagName]),
                  removed: row.rule.removed.filter((x) => x !== tagName),
                })
              }
              onClose={() => setTagEdit(null)}
            />
          ) : (
            <span style={addChip} onClick={() => setTagEdit(node.rel_path)}>
              + tag
            </span>
          )}
        </div>
      )}

      {isSub && (
        <div style={subLine}>
          <span style={{ color: colors.accent, fontFamily: font.mono }}>
            ◫ library name
          </span>
          <input
            value={row.subName}
            onChange={(e) =>
              patch(node.rel_path, node, { subName: e.target.value })
            }
            style={{
              ...smallInput,
              width: 160,
              color: colors.accent,
              border: `1px solid ${colors.accentBorder}`,
              background: "#0f1013",
            }}
          />
          <span style={{ ...crumb, fontSize: 9.5 }}>
            excluded from “{row.owner}” — its media appear only here
          </span>
        </div>
      )}
    </div>
  );
}

function dedupe(list: string[]): string[] {
  return Array.from(new Set(list));
}

function FolderTagInput({
  current,
  localTags,
  onAdd,
  onClose,
}: {
  current: string[];
  /** Tag names created elsewhere in this draft (not yet persisted). */
  localTags: string[];
  onAdd: (name: string) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const trimmed = query.trim();
  const search = useTagSearch(trimmed, trimmed.length > 0);
  // Free-form: spaces are kept verbatim (no slugging on manual entry).
  const typed = trimmed;
  const chosen = new Set(current);
  const lower = trimmed.toLowerCase();
  const serverNames = new Set(
    (search.data?.tags ?? []).map((tag) => tag.name),
  );
  // Tags created earlier in this same wizard aren't persisted yet, so the
  // server search can't return them — surface them from the draft too.
  const localMatches = localTags.filter(
    (tag) =>
      tag.toLowerCase().includes(lower) &&
      !chosen.has(tag) &&
      !serverNames.has(tag),
  );
  const results = [
    ...localMatches.map((name) => ({ id: `local:${name}`, name })),
    ...(search.data?.tags ?? []).filter((tag) => !chosen.has(tag.name)),
  ].slice(0, 6);
  const exact =
    localTags.some((tag) => tag.toLowerCase() === lower) ||
    (search.data?.tags ?? []).some((tag) => tag.name.toLowerCase() === lower);
  const canCreate = !!typed && !exact && !chosen.has(typed);
  const commit = (name: string) => {
    if (name) onAdd(name);
    setQuery("");
  };

  return (
    <span style={{ position: "relative", display: "inline-flex" }}>
      <input
        autoFocus
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit(typed);
          else if (e.key === "Escape") onClose();
        }}
        // Delay so a dropdown mousedown commits before blur closes the input.
        onBlur={() => setTimeout(onClose, 120)}
        placeholder="tag"
        style={{
          ...smallInput,
          width: 110,
          border: `1px solid ${
            typed && !exact ? colors.accentBorder : colors.borderControl
          }`,
        }}
      />
      {typed && !exact && (
        <span
          style={{
            ...crumb,
            fontSize: 9.5,
            color: colors.warn,
            alignSelf: "center",
            marginLeft: 4,
          }}
        >
          ↵ creates new tag
        </span>
      )}
      {trimmed && (results.length > 0 || canCreate) && (
        <div style={acDropdown}>
          {results.map((tag) => (
            <div
              key={tag.id}
              onMouseDown={(e) => {
                e.preventDefault();
                commit(tag.name);
              }}
              style={acItem}
            >
              {tag.name}
            </div>
          ))}
          {canCreate && (
            <div
              onMouseDown={(e) => {
                e.preventDefault();
                commit(typed);
              }}
              style={{ ...acItem, color: colors.accent }}
            >
              ➕ create “{typed}”
            </div>
          )}
        </div>
      )}
    </span>
  );
}

function Chip({
  label,
  color,
  bg,
  borderColor,
  borderStyle = "solid",
  title,
  onRemove,
}: {
  label: string;
  color: string;
  bg: string;
  borderColor: string;
  borderStyle?: "solid" | "dashed";
  title?: string;
  onRemove: () => void;
}) {
  return (
    <span
      title={title}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        fontFamily: font.mono,
        fontSize: 10,
        padding: "1px 7px",
        borderRadius: 9,
        color,
        background: bg,
        border: `1px ${borderStyle} ${borderColor}`,
      }}
    >
      {label}
      <span
        onClick={onRemove}
        style={{ cursor: "pointer", color: colors.textFaint }}
        onMouseEnter={(e) => (e.currentTarget.style.color = colors.danger)}
        onMouseLeave={(e) => (e.currentTarget.style.color = colors.textFaint)}
      >
        ×
      </span>
    </span>
  );
}

function SummaryRail({
  libs,
  excludedFiles,
  taggedFiles,
  ruleCount,
}: {
  libs: Resolved["libs"];
  excludedFiles: number;
  taggedFiles: number;
  ruleCount: number;
}) {
  return (
    <div style={rail}>
      <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 8 }}>
        Resulting libraries
      </div>
      {libs.map((lib) => (
        <div
          key={`${lib.sub ? "s" : "p"}-${lib.name}`}
          style={{
            ...railCard,
            marginLeft: lib.sub ? 14 : 0,
            borderColor: lib.sub ? colors.accentBorder : colors.border,
          }}
        >
          <div
            style={{
              fontSize: 11.5,
              fontWeight: 600,
              color: lib.sub ? colors.accent : colors.text,
            }}
          >
            {lib.sub ? "└ " : ""}
            {lib.name}
          </div>
          {lib.sub && (
            <div style={{ ...crumb, fontSize: 9 }}>sub-library of {lib.parent}</div>
          )}
          <div style={{ ...crumb, fontSize: 9.5 }}>{lib.count} files</div>
        </div>
      ))}
      {excludedFiles > 0 && (
        <div style={skippedBanner}>
          ⊘ {excludedFiles} files skipped — never scanned
        </div>
      )}
      <div style={{ ...crumb, fontSize: 9.5, marginTop: 10 }}>
        {ruleCount} folder rules → {taggedFiles} files pre-tagged at scan
      </div>
      <div style={{ fontSize: 9.5, color: colors.textFaint, marginTop: 6 }}>
        Rules are persistent — files added later to a mapped folder inherit its
        tags and library automatically.
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ atoms */

function ModalHeader({
  title,
  subtitle,
  onClose,
}: {
  title: string;
  subtitle: string;
  onClose: () => void;
}) {
  return (
    <div style={mapHeader}>
      <span style={{ fontSize: 14, fontWeight: 600 }}>{title}</span>
      <span style={{ ...crumb, flex: 1 }}>{subtitle}</span>
      <span style={closeX} onClick={onClose}>
        ✕
      </span>
    </div>
  );
}

const legendChip = (
  color: string,
  bg: string,
  borderStyle: string,
  borderColor: string,
) =>
  ({
    fontFamily: font.mono,
    fontSize: 9.5,
    padding: "1px 7px",
    borderRadius: 9,
    color,
    background: bg,
    border: `1px ${borderStyle} ${borderColor}`,
  }) as const;

/* ------------------------------------------------------------------ style */

const backdrop = {
  position: "fixed",
  inset: 0,
  background: "rgba(0,0,0,0.55)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 60,
} as const;

const basePanel = {
  display: "flex",
  flexDirection: "column",
  background: "#15161b",
  border: `1px solid ${colors.border}`,
  borderRadius: radii.modal,
  boxShadow: shadow.modal,
  overflow: "hidden",
  maxHeight: "88vh",
} as const;

const pickPanel = { ...basePanel, width: 560, maxWidth: "92vw" } as const;
const mapPanel = { ...basePanel, width: 940, maxWidth: "96vw" } as const;

const mapHeader = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "12px 14px",
  borderBottom: `1px solid ${colors.border}`,
} as const;

const closeX = { cursor: "pointer", color: colors.textMuted } as const;
const backBtn = {
  cursor: "pointer",
  color: colors.textMuted,
  fontSize: 12,
} as const;

const crumb = {
  fontFamily: font.mono,
  fontSize: 11,
  color: colors.textMuted,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} as const;

const explainer = {
  padding: "8px 14px",
  fontSize: 11,
  color: colors.textMuted,
  borderBottom: `1px solid ${colors.border}`,
} as const;

const autoToolbar = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "8px 14px",
  background: "#131418",
  borderBottom: `1px solid ${colors.border}`,
  flexWrap: "wrap",
} as const;

const segGroup = {
  display: "flex",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: 6,
  overflow: "hidden",
} as const;

const segItem = {
  padding: "3px 10px",
  fontFamily: font.mono,
  fontSize: 10.5,
  cursor: "pointer",
} as const;

const bulkBar = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "7px 14px",
  background: "#1f1a11",
  borderBottom: `1px solid ${colors.accentBorder}`,
} as const;

const mapBody = {
  display: "flex",
  gap: 12,
  padding: 12,
  minHeight: 0,
  overflow: "hidden",
} as const;

const treePane = {
  flex: 1,
  minWidth: 0,
  overflowY: "auto",
  display: "flex",
  flexDirection: "column",
  gap: 6,
} as const;

const treeHeader = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "0 10px 4px 0",
} as const;

const modeGroupBox = {
  display: "flex",
  border: `1px solid ${colors.borderControl}`,
  borderRadius: 6,
  overflow: "hidden",
} as const;

const rowBox = {
  borderRadius: 8,
  padding: "8px 10px",
  display: "flex",
  flexDirection: "column",
  gap: 6,
} as const;

const rootRowBox = {
  borderRadius: 8,
  padding: "8px 10px",
  display: "flex",
  flexDirection: "column",
  gap: 6,
  border: "1px solid #3a3324",
  background: "#191712",
} as const;

const rootBadge = {
  fontFamily: font.mono,
  fontSize: 9,
  padding: "0 6px",
  borderRadius: 9,
  color: colors.accent,
  background: colors.accentTint,
  border: `1px solid ${colors.accentBorder}`,
} as const;

const searchBar = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "6px 8px",
  borderRadius: 8,
  border: `1px solid ${colors.border}`,
  background: "#131418",
} as const;

const searchInput = {
  flex: 1,
  minWidth: 0,
  padding: "3px 4px",
  border: "none",
  background: "transparent",
  color: colors.text,
  fontSize: 12,
  outline: "none",
} as const;

const pickSearchWrap = {
  padding: "8px 14px 0",
} as const;

const thumbMini = {
  width: 24,
  height: 24,
  borderRadius: 4,
  background: "#22242b",
  border: `1px solid ${colors.border}`,
} as const;

const chipsZone = {
  display: "flex",
  flexWrap: "wrap",
  gap: 5,
  alignItems: "center",
  paddingLeft: 32,
} as const;

const addChip = {
  fontFamily: font.mono,
  fontSize: 10,
  padding: "1px 7px",
  borderRadius: 9,
  color: colors.textMuted,
  border: `1px dashed ${colors.borderControl}`,
  cursor: "pointer",
} as const;

const subLine = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  paddingLeft: 32,
  fontSize: 9.5,
} as const;

const rail = {
  width: 252,
  flex: "none",
  background: "#131418",
  border: `1px solid ${colors.border}`,
  borderRadius: 8,
  padding: 12,
  overflowY: "auto",
} as const;

const railCard = {
  border: `1px solid ${colors.border}`,
  borderRadius: 7,
  padding: "7px 9px",
  marginBottom: 6,
} as const;

const skippedBanner = {
  marginTop: 8,
  padding: "6px 9px",
  borderRadius: 6,
  background: "#1b1214",
  border: "1px solid #3a2622",
  color: colors.danger,
  fontFamily: font.mono,
  fontSize: 10,
} as const;

const footer = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "10px 14px",
  borderTop: `1px solid ${colors.border}`,
} as const;

const pickToolbar = {
  display: "flex",
  alignItems: "center",
  gap: 6,
  padding: "8px 14px",
  borderBottom: `1px solid ${colors.border}`,
} as const;

const drivePill = {
  padding: "3px 10px",
  borderRadius: 6,
  fontFamily: font.mono,
  fontSize: 11,
  cursor: "pointer",
} as const;

const pickBody = {
  flex: 1,
  overflowY: "auto",
  padding: 6,
  minHeight: 200,
} as const;

const pickRow = {
  display: "flex",
  alignItems: "center",
  gap: 9,
  padding: "7px 10px",
  borderRadius: 6,
  cursor: "pointer",
} as const;

const smallInput = {
  padding: "4px 7px",
  borderRadius: 5,
  border: `1px solid ${colors.borderControl}`,
  background: colors.input,
  color: colors.text,
  fontSize: 11,
  outline: "none",
} as const;

const linkBtn = {
  fontSize: 11,
  color: colors.textMuted,
  cursor: "pointer",
} as const;

const acDropdown = {
  position: "absolute",
  top: "100%",
  left: 0,
  zIndex: 30,
  marginTop: 2,
  minWidth: 130,
  maxHeight: 180,
  overflowY: "auto",
  background: colors.panel,
  border: `1px solid ${colors.borderHover}`,
  borderRadius: 6,
} as const;

const acItem = {
  padding: "5px 9px",
  fontSize: 11.5,
  cursor: "pointer",
  color: colors.textSecondary,
  fontFamily: font.mono,
} as const;
