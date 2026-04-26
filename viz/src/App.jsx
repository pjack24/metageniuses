import { HashRouter, Routes, Route } from "react-router-dom";
import LandingPage from "./LandingPage";
import ExplorerView from "./ExplorerView";
import ExperimentsLayout from "./pages/ExperimentsLayout";
import Experiment1 from "./pages/Experiment1";
import Experiment2 from "./pages/Experiment2";
import Experiment3 from "./pages/Experiment3";
import Experiment4 from "./pages/Experiment4";
import Experiment5 from "./pages/Experiment5";

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/explorer" element={<ExplorerView />} />
        <Route path="/experiments" element={<ExperimentsLayout />}>
          <Route index element={<Experiment1 />} />
          <Route path="1" element={<Experiment1 />} />
          <Route path="2" element={<Experiment2 />} />
          <Route path="3" element={<Experiment3 />} />
          <Route path="4" element={<Experiment4 />} />
          <Route path="5" element={<Experiment5 />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}
