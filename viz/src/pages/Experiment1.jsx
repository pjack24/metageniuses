import useApi from "../hooks/useApi";
import { Loading, ErrorState } from "../components/LoadingState";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell,
} from "recharts";

const COLORS = { pathogen: "#8a0038", nonpathogen: "#0d8ba1", ns: "#d4d4d4" };

export default function Experiment1() {
  const { data, loading, error } = useApi("/api/experiments/1");

  if (loading) return <Loading />;
  if (error) return <ErrorState message={error} />;

  const { summary, volcano, top_detectors, enrichment_histogram } = data;

  return (
    <div className="space-y-6">
      {/* Title + summary */}
      <div>
        <h2 className="text-2xl mb-1">Organism-Specific Pathogen Detectors</h2>
        <p className="text-sm text-[#828282]" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          SAE latents that fire specifically on pathogen sequences, identified via enrichment analysis + BLAST.
        </p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          { label: "Total Latents", value: summary.total_latents.toLocaleString() },
          { label: "Alive", value: summary.alive_latents.toLocaleString() },
          { label: "Pathogen-Enriched", value: summary.pathogen_enriched },
          { label: "Non-Pathogen-Enriched", value: summary.nonpathogen_enriched },
          { label: "High F1 (>0.7)", value: summary.high_f1_latents },
        ].map((s) => (
          <div key={s.label} className="bg-white rounded-lg border border-gray-100 p-3 text-center shadow-sm">
            <div className="text-xl text-[#0d8ba1]" style={{ fontFamily: "'VT323', monospace" }}>{s.value}</div>
            <div className="text-xs text-[#828282]">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Volcano plot */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
          Volcano Plot
        </h3>
        <p className="text-xs text-[#828282] mb-3" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          Each point is one SAE latent. X: log2 fold-change (pathogen vs non-pathogen). Y: -log10 FDR p-value.
        </p>
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis type="number" dataKey="log2fc" name="log2FC" domain={[-5, 5]}
              label={{ value: "log2 Fold Change", position: "bottom", offset: 15, style: { fontSize: 12 } }} />
            <YAxis type="number" dataKey="neg_log10_pval" name="-log10(p)"
              label={{ value: "-log10(FDR p-value)", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12 } }} />
            <Tooltip content={<VolcanoTooltip />} />
            {["ns", "nonpathogen", "pathogen"].map((dir) => (
              <Scatter
                key={dir}
                data={volcano.filter((v) => v.direction === dir)}
                fill={COLORS[dir]}
                fillOpacity={dir === "ns" ? 0.3 : 0.7}
                r={dir === "ns" ? 2 : 3}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
        <div className="flex items-center justify-center gap-6 mt-2 text-xs text-[#828282]">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#8a0038] inline-block" />Pathogen-enriched</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#0d8ba1] inline-block" />Non-pathogen-enriched</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#d4d4d4] inline-block" />Not significant</span>
        </div>
      </div>

      {/* Top organism detectors table */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
          Top Organism Detectors
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 text-left text-xs text-[#828282]">
                <th className="py-2 pr-3">Latent</th>
                <th className="py-2 pr-3">Enrichment</th>
                <th className="py-2 pr-3">FDR p-val</th>
                <th className="py-2 pr-3">Max F1</th>
                <th className="py-2 pr-3">BLAST Hits</th>
                <th className="py-2">Proposed Label</th>
              </tr>
            </thead>
            <tbody>
              {top_detectors.map((d) => (
                <tr key={d.latent_id} className="border-b border-gray-50 hover:bg-gray-50">
                  <td className="py-2 pr-3 font-mono text-xs">#{d.latent_id}</td>
                  <td className="py-2 pr-3">{d.odds_ratio}x</td>
                  <td className="py-2 pr-3 font-mono text-xs">{d.fdr_pval.toExponential(1)}</td>
                  <td className="py-2 pr-3">
                    <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${d.max_f1 >= 0.7 ? "bg-[#4e8c02]/10 text-[#4e8c02]" : "bg-gray-100 text-gray-500"}`}>
                      {d.max_f1.toFixed(2)}
                    </span>
                  </td>
                  <td className="py-2 pr-3 text-xs">{d.hit_consistency}</td>
                  <td className="py-2 font-bold text-[#8a0038]">{d.proposed_label}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Enrichment distribution */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
          Odds Ratio Distribution
        </h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={enrichment_histogram}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="bin_start" tick={{ fontSize: 11 }} label={{ value: "Odds Ratio", position: "bottom", offset: 10, style: { fontSize: 12 } }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#0d8ba1" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function VolcanoTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-md shadow-sm p-2 text-xs">
      <p className="font-bold">Latent #{d.latent_id}</p>
      <p>log2FC: {d.log2fc}</p>
      <p>-log10(p): {d.neg_log10_pval}</p>
      <p className="capitalize">{d.direction}</p>
    </div>
  );
}
