import { useEffect, useState } from "react";
import ScanCard from "../components/ScanCard";
import Loader from "../components/Loader";
import { analyzeScan, getPendingScans } from "../services/scanService";
import { extractApiErrorMessage } from "../utils/helpers";

function PendingScans() {
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [analyzingId, setAnalyzingId] = useState(null);
  const [message, setMessage] = useState("");

  const fetchPendingScans = async () => {
    try {
      setLoading(true);
      setMessage("");
      const response = await getPendingScans();
      setScans(response.data || []);
    } catch (error) {
      console.error("Failed to fetch pending scans:", error?.response?.data || error.message);
      setMessage(extractApiErrorMessage(error, "Failed to fetch pending scans."));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPendingScans();
  }, []);

  const handleAnalyze = async (id) => {
    try {
      setAnalyzingId(id);
      setMessage("");
      const response = await analyzeScan(id);
      const prediction = response?.data?.prediction || "updated";
      setMessage(`Scan #${id} analyzed successfully. Result: ${prediction}`);
      await fetchPendingScans();
    } catch (error) {
      console.error("Analyze failed:", error?.response?.data || error.message);
      setMessage(extractApiErrorMessage(error, "Analyze failed."));
    } finally {
      setAnalyzingId(null);
    }
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Pending Scans</h1>
          <p className="section-text">
            Review and analyze scans that are still waiting for prediction.
          </p>
        </div>
      </div>

      {message && <div className="message-box">{message}</div>}

      {loading ? (
        <Loader text="Loading pending scans..." />
      ) : scans.length === 0 ? (
        <div className="empty-panel">
          <h2>No pending scans</h2>
          <p className="section-text">
            All uploaded scans are currently processed.
          </p>
        </div>
      ) : (
        <div className="scan-list">
          {scans.map((scan) => (
            <ScanCard
              key={scan.id}
              scan={scan}
              onAnalyze={handleAnalyze}
              analyzing={analyzingId === scan.id}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default PendingScans;