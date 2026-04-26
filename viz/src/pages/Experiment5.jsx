import { useState } from "react";
import useApi from "../hooks/useApi";
import { Loading, ErrorState } from "../components/LoadingState";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

const CLUSTER_COLORS = ["#8a0038", "#0d8ba1", "#4e8c02", "#e67e22", "#9b59b6", "#34495e", "#828282"];

export default function Experiment5() {
  const { data, loading, error } = useApi("/api/experiments/5");
  const [colorBy, setColorBy] = useState("cluster"); // "cluster" | "enrichment"

  if (loading) return <Loading />;
  if (error) return <ErrorState message={error} />;

  const { points, cluster_summary, summary } = data;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl mb-1">Feature Clustering</h2>
        <p className="text-sm text-[#828282]" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          32,768 SAE latents projected to 2D via UMAP (cosine distance on co-activation patterns), clustered with HDBSCAN.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "Total Latents", value: summary.n_latents.toLocaleString() },
          { label: "Clusters Found", value: summary.n_clusters },
          { label: "Noise Points", value: summary.noise_count },
        ].map((s) => (
          <div key={s.label} className="bg-white rounded-lg border border-gray-100 p-3 text-center shadow-sm">
            <div className="text-xl text-[#0d8ba1]" style={{ fontFamily: "'VT323', monospace" }}>{s.value}</div>
            <div className="text-xs text-[#828282]">{s.label}</div>
          </div>
        ))}
      </div>

      {/* UMAP scatter with toggle */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-700" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
            Latent UMAP
          </h3>
          <div className="flex gap-1">
            {[
              { key: "cluster", label: "By Cluster" },
              { key: "enrichment", label: "By Enrichment" },
            ].map((opt) => (
              <button
                key={opt.key}
                onClick={() => setColorBy(opt.key)}
                className={`px-3 py-1 text-xs rounded-md border cursor-pointer transition ${
                  colorBy === opt.key
                    ? "bg-[#0d8ba1] text-white border-[#0d8ba1]"
                    : "bg-white text-[#828282] border-gray-200 hover:bg-gray-50"
                }`}
                style={{ fontFamily: "'Roboto Condensed', sans-serif" }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={500}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis type="number" dataKey="x" name="UMAP 1"
              label={{ value: "UMAP 1", position: "bottom", offset: 15, style: { fontSize: 12 } }} />
            <YAxis type="number" dataKey="y" name="UMAP 2"
              label={{ value: "UMAP 2", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12 } }} />
            <Tooltip content={<ClusterTooltip />} />
            <Scatter data={points} fillOpacity={0.5} r={2}>
              {points.map((p, i) => (
                <Cell
                  key={i}
                  fill={
                    colorBy === "cluster"
                      ? p.cluster_id === -1
                        ? "#d4d4d4"
                        : CLUSTER_COLORS[p.cluster_id % CLUSTER_COLORS.length]
                      : enrichmentColor(p.enrichment)
                  }
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>

        {/* Legend */}
        {colorBy === "cluster" ? (
          <div className="flex flex-wrap items-center justify-center gap-4 mt-2 text-xs text-[#828282]">
            {cluster_summary.map((c) => (
              <span key={c.cluster_id} className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full inline-block" style={{ backgroundColor: CLUSTER_COLORS[c.cluster_id % CLUSTER_COLORS.length] }} />
                {c.label}
              </span>
            ))}
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#d4d4d4] inline-block" />Noise</span>
          </div>
        ) : (
          <div className="flex items-center justify-center gap-2 mt-2 text-xs text-[#828282]">
            <span>Non-pathogen</span>
            <div className="w-32 h-3 rounded" style={{ background: "linear-gradient(to right, #0d8ba1, #d4d4d4, #8a0038)" }} />
            <span>Pathogen</span>
          </div>
        )}
      </div>

      {/* Cluster summary table */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
          Cluster Summary
        </h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 text-left text-xs text-[#828282]">
              <th className="py-2 pr-3">Cluster</th>
              <th className="py-2 pr-3">Label</th>
              <th className="py-2 pr-3">Size</th>
              <th className="py-2 pr-3">Mean Enrichment</th>
              <th className="py-2">Mean Act. Count</th>
            </tr>
          </thead>
          <tbody>
            {cluster_summary.map((c) => (
              <tr key={c.cluster_id} className="border-b border-gray-50 hover:bg-gray-50">
                <td className="py-2 pr-3">
                  <span className="inline-block w-3 h-3 rounded-full mr-2" style={{ backgroundColor: CLUSTER_COLORS[c.cluster_id % CLUSTER_COLORS.length] }} />
                  {c.cluster_id}
                </td>
                <td className="py-2 pr-3 font-bold">{c.label}</td>
                <td className="py-2 pr-3">{c.size}</td>
                <td className="py-2 pr-3">
                  <span className={c.mean_enrichment > 2 ? "text-[#8a0038] font-bold" : c.mean_enrichment < 0.5 ? "text-[#0d8ba1] font-bold" : ""}>
                    {c.mean_enrichment.toFixed(2)}x
                  </span>
                </td>
                <td className="py-2">{c.mean_activation_count.toFixed(0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function enrichmentColor(enrichment) {
  if (enrichment > 2.5) return "#8a0038";
  if (enrichment > 1.5) return "#c0392b";
  if (enrichment < 0.4) return "#0d8ba1";
  if (enrichment < 0.7) return "#2980b9";
  return "#d4d4d4";
}

function ClusterTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-md shadow-sm p-2 text-xs">
      <p className="font-bold">Latent #{d.latent_id}</p>
      <p>Cluster: {d.cluster_id === -1 ? "Noise" : d.cluster_id}</p>
      <p>Enrichment: {d.enrichment}x</p>
      <p>Act. count: {d.activation_count}</p>
    </div>
  );
}
