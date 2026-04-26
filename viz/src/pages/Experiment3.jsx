import useApi from "../hooks/useApi";
import { Loading, ErrorState } from "../components/LoadingState";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

export default function Experiment3() {
  const { data, loading, error } = useApi("/api/experiments/3");

  if (loading) return <Loading />;
  if (error) return <ErrorState message={error} />;

  const { summary, sequences_per_latent, max_activation_dist, active_features_per_seq, comparison } = data;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl mb-1">SAE Health Check</h2>
        <p className="text-sm text-[#828282]" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>
          Descriptive statistics on the trained SAE. Dead/alive census, activation distributions, sparsity.
        </p>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: "Total Latents", value: summary.total_latents.toLocaleString() },
          { label: "Alive", value: summary.alive_count.toLocaleString(), color: "#4e8c02" },
          { label: "Dead", value: summary.dead_count.toLocaleString(), color: "#8a0038" },
          { label: "Dead %", value: summary.dead_pct + "%" },
          { label: "Sparsity", value: summary.sparsity_pct + "%" },
          { label: "Mean Active/Seq", value: summary.mean_active_per_seq },
          { label: "Median Active/Seq", value: summary.median_active_per_seq },
          { label: "Mean Act. Count", value: summary.mean_activation_count },
        ].map((s) => (
          <div key={s.label} className="bg-white rounded-lg border border-gray-100 p-3 text-center shadow-sm">
            <div className="text-xl" style={{ fontFamily: "'VT323', monospace", color: s.color || "#0d8ba1" }}>{s.value}</div>
            <div className="text-xs text-[#828282]">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Histograms */}
      <div className="grid md:grid-cols-2 gap-6">
        <HistCard
          title="Sequences per Latent"
          subtitle="How many sequences activate each latent (expect long tail)"
          data={sequences_per_latent}
          dataKey="count"
          xKey="bin_start"
          xLabel="# Sequences"
        />
        <HistCard
          title="Max Activation Distribution"
          subtitle="Distribution of peak activation values across latents"
          data={max_activation_dist}
          dataKey="count"
          xKey="bin_start"
          xLabel="Max Activation"
        />
      </div>

      <HistCard
        title="Active Features per Sequence"
        subtitle="Should peak near k=64 (TopK sparsity)"
        data={active_features_per_seq}
        dataKey="count"
        xKey="bin_center"
        xLabel="# Active Features"
        wide
      />

      {/* Comparison table */}
      <div className="bg-white rounded-lg border border-gray-100 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-3" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
          Comparison to InterProt
        </h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 text-left text-xs text-[#828282]">
              <th className="py-2 pr-3">Metric</th>
              <th className="py-2 pr-3">InterProt (ESM-2)</th>
              <th className="py-2">Ours (MetaGene-1)</th>
            </tr>
          </thead>
          <tbody>
            {["d_model", "expansion", "k", "total_latents", "dead_pct"].map((key) => (
              <tr key={key} className="border-b border-gray-50">
                <td className="py-2 pr-3 text-[#828282]">{key}</td>
                <td className="py-2 pr-3">{String(comparison.interprot[key])}</td>
                <td className="py-2 font-bold">{String(comparison.ours[key])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function HistCard({ title, subtitle, data, dataKey, xKey, xLabel, wide }) {
  return (
    <div className={`bg-white rounded-lg border border-gray-100 p-5 shadow-sm ${wide ? "" : ""}`}>
      <h3 className="text-sm font-semibold text-gray-700 mb-1" style={{ fontFamily: "'VT323', monospace", textTransform: "uppercase", fontSize: "1rem" }}>
        {title}
      </h3>
      <p className="text-xs text-[#828282] mb-3" style={{ fontFamily: "'Roboto Condensed', sans-serif" }}>{subtitle}</p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey={xKey} tick={{ fontSize: 10 }} label={{ value: xLabel, position: "bottom", offset: 10, style: { fontSize: 12 } }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Bar dataKey={dataKey} fill="#0d8ba1" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
