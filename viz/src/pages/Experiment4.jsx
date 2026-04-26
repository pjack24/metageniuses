import useApi from "../hooks/useApi";
import { Loading, ErrorState } from "../components/LoadingState";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar,
} from "recharts";

export default function Experiment4() {
  const { data, loading, error } = useApi("/api/experiments/4");

  if (loading) return <Loading />;
  if (error) return <ErrorState message={error} />;

  const { points, pca_variance, summary } = data;

  const pathogenPoints = points.filter((p) => p.label === 1);
  const nonpathogenPoints = points.filter((p) => p.label === 0);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl mb-1">Sequence UMAP</h2>
        <p className="text-sm text-[#828282]" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          20,000 sequences projected from 32,768-dim SAE space to 2D via PCA(50) + UMAP.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: "Sequences", value: summary.n_sequences.toLocaleString() },
          { label: "Pathogen", value: summary.n_pathogen.toLocaleString(), color: "#8a0038" },
          { label: "Non-Pathogen", value: summary.n_nonpathogen.toLocaleString(), color: "#0d8ba1" },
          { label: "Variance (50 PCs)", value: (summary.variance_explained_50 * 100).toFixed(1) + "%" },
        ].map((s) => (
          <div key={s.label} className="bg-white rounded-lg border border-gray-100 p-3 text-center shadow-sm">
            <div className="text-xl" style={{ fontFamily: "'VT323', monospace", color: s.color || "#0d8ba1" }}>{s.value}</div>
            <div className="text-xs text-[#828282]">{s.label}</div>
          </div>
        ))}
      </div>

      {/* UMAP scatter */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
          UMAP Projection
        </h3>
        <ResponsiveContainer width="100%" height={500}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis type="number" dataKey="x" name="UMAP 1"
              label={{ value: "UMAP 1", position: "bottom", offset: 15, style: { fontSize: 12 } }} />
            <YAxis type="number" dataKey="y" name="UMAP 2"
              label={{ value: "UMAP 2", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12 } }} />
            <Tooltip content={<UmapTooltip />} />
            <Scatter name="Non-pathogen" data={nonpathogenPoints} fill="#0d8ba1" fillOpacity={0.3} r={2} />
            <Scatter name="Pathogen" data={pathogenPoints} fill="#8a0038" fillOpacity={0.3} r={2} />
          </ScatterChart>
        </ResponsiveContainer>
        <div className="flex items-center justify-center gap-6 mt-2 text-xs text-[#828282]">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#8a0038] inline-block" />Pathogen</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#0d8ba1] inline-block" />Non-pathogen</span>
        </div>
      </div>

      {/* PCA variance scree plot */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
          PCA Scree Plot
        </h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={pca_variance}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="component" tick={{ fontSize: 10 }} label={{ value: "Principal Component", position: "bottom", offset: 10, style: { fontSize: 12 } }} />
            <YAxis tick={{ fontSize: 11 }} label={{ value: "Variance Explained", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12 } }} />
            <Tooltip />
            <Bar dataKey="explained_variance" fill="#0d8ba1" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function UmapTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-md shadow-sm p-2 text-xs">
      <p className="font-bold">{d.sequence_id}</p>
      <p>{d.label === 1 ? "Pathogen" : "Non-pathogen"}</p>
    </div>
  );
}
