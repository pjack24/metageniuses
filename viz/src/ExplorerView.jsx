import { useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import Sidebar from "./Sidebar";
import FeaturePanel from "./FeaturePanel";
import DetailsPanel from "./DetailsPanel";
import useApi from "./hooks/useApi";
import { Loading, ErrorState } from "./components/LoadingState";

export default function ExplorerView() {
  const { data: features, loading, error } = useApi("/api/features");
  const [selectedFeature, setSelectedFeature] = useState(null);

  if (loading) return <div className="h-screen flex items-center justify-center bg-gray-100"><Loading /></div>;
  if (error) return <div className="h-screen flex items-center justify-center bg-gray-100"><ErrorState message={error} /></div>;

  const active = selectedFeature || features[0];

  return (
    <div className="h-screen flex flex-col bg-gray-100 text-gray-900">
      <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-200 bg-white">
        <Link
          to="/"
          className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-gray-200 bg-white hover:bg-gray-50 transition no-underline"
        >
          <ArrowLeft className="w-4 h-4 text-gray-500" />
          <span className="text-sm text-gray-600">Back</span>
        </Link>
        <h1 className="text-lg tracking-wide">
          Meta<span className="text-[#0d8ba1]">Geniuses</span>
          <span className="text-[#828282] ml-2 text-base" style={{ fontFamily: "'Roboto Condensed', sans-serif", textTransform: "none" }}>Feature Explorer</span>
        </h1>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          features={features}
          selected={active}
          onSelect={setSelectedFeature}
        />
        <FeaturePanel feature={active} />
        <DetailsPanel feature={active} />
      </div>
    </div>
  );
}
