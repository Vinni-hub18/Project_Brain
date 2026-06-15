import { Route, Routes } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import Upload from "./pages/Upload";
import Scans from "./pages/Scans";
import PendingScans from "./pages/PendingScans";
import ScanDetails from "./pages/ScanDetails";
import Chatbot from "./pages/Chatbot";

function App() {
  return (
    <div className="app-shell">
      <Sidebar />

      <div className="app-main">
        <Navbar />

        <main className="page-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/scans" element={<Scans />} />
            <Route path="/pending" element={<PendingScans />} />
            <Route path="/scans/:id" element={<ScanDetails />} />
            <Route path="/chatbot" element={<Chatbot />} />
            <Route
              path="*"
              element={
                <div className="empty-panel">
                  <h1>Page not found</h1>
                  <p className="section-text">
                    The page you are trying to access does not exist.
                  </p>
                </div>
              }
            />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default App;