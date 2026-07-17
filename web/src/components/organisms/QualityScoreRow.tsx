/**
 * The Quality report's score row: the global gauge plus the three pillar
 * cards (image quality, diversity, hygiene). Every number comes from the
 * stored report — nothing is recomputed here.
 */

import { colors, font, qualityColor, radii, toneColor } from "../../design/tokens";
import type { DatasetReport, ReportPillar } from "../../api/types";

const cardStyle = {
  background: colors.card,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.card,
  padding: 14,
} as const;

function weightNote(weights: Record<string, number>): string {
  const order = ["quality", "diversity", "composition", "hygiene"];
  return order
    .filter((key) => weights[key] != null)
    .map((key) => `${key} ${Math.round(weights[key] * 100)}%`)
    .join(" · ");
}

function Gauge({ score, grade }: { score: number | null; grade: string }) {
  const value = score ?? 0;
  const color = qualityColor(score);
  return (
    <div
      style={{
        width: 128,
        height: 128,
        borderRadius: "50%",
        display: "grid",
        placeItems: "center",
        background: `conic-gradient(${color} ${value}%, ${colors.border} 0)`,
      }}
    >
      <div
        style={{
          width: 102,
          height: 102,
          borderRadius: "50%",
          background: colors.card,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 2,
        }}
      >
        <span style={{ fontSize: 32, fontWeight: 700, color }}>
          {score == null ? "—" : Math.round(score)}
        </span>
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 10,
            color: colors.textFaint,
          }}
        >
          / 100 · grade {grade}
        </span>
      </div>
    </div>
  );
}

function PillarCard({ pillar }: { pillar: ReportPillar }) {
  return (
    <div style={cardStyle}>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          marginBottom: 4,
        }}
      >
        <span style={{ fontSize: 12.5, fontWeight: 600 }}>{pillar.label}</span>
        <span
          style={{
            fontFamily: font.mono,
            fontSize: 18,
            fontWeight: 700,
            color: qualityColor(pillar.score),
          }}
        >
          {pillar.score == null ? "—" : Math.round(pillar.score)}
        </span>
      </div>
      <div style={{ fontSize: 11, color: colors.textFaint, marginBottom: 10 }}>
        {pillar.detail}
      </div>
      {pillar.rows.map((row) => (
        <div
          key={row.label}
          style={{
            display: "flex",
            justifyContent: "space-between",
            padding: "3px 0",
            fontSize: 11.5,
            color: colors.textMuted,
          }}
        >
          <span>{row.label}</span>
          <span
            style={{
              fontFamily: font.mono,
              color: toneColor(row.tone),
            }}
          >
            {row.value}
          </span>
        </div>
      ))}
    </div>
  );
}

export function QualityScoreRow({ report }: { report: DatasetReport }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "264px 1fr", gap: 12 }}>
      <div style={{ ...cardStyle, display: "flex", gap: 14 }}>
        <Gauge score={report.overall} grade={report.grade} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontSize: 12.5,
              fontWeight: 600,
              color: qualityColor(report.overall),
            }}
          >
            {report.verdict}
          </div>
          <div
            style={{
              fontSize: 11,
              color: colors.textMuted,
              margin: "6px 0 10px",
            }}
          >
            {report.summary}
          </div>
          <div
            style={{
              fontFamily: font.mono,
              fontSize: 9.5,
              color: colors.textFaint,
            }}
          >
            {weightNote(report.weights)}
          </div>
        </div>
      </div>
      {/* Four pillars in a 2×2 grid: quality | diversity on top, composition
          | hygiene below — the extra card no longer fits the old single row. */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {report.pillars.map((pillar) => (
          <PillarCard key={pillar.key} pillar={pillar} />
        ))}
      </div>
    </div>
  );
}
