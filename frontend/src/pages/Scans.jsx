import { useEffect, useState } from "react";
import {
  getAllScans,
  analyzeScan,
  getPredictionColor,
  getPredictionLabel,
} from "../services/scanService";

function Scans() {
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [actionMessage, setActionMessage] = useState("");
  const [analyzingId, setAnalyzingId] = useState(null);

  const fetchScans = async () => {
    try {
      setLoading(true);
      const response = await getAllScans();
      setScans(response.data || []);
    } catch (error) {
      console.error("Failed to fetch scans", error?.response?.data || error.message);
      setActionMessage("Failed to fetch scans.");
    } finally {
      setLoading(false);
    }
  };

  const extractAnalyzeMessage = (data) => {
    if (!data) return "Analysis completed.";
    if (data.validation_message) return data.validation_message;
    if (data.insight_text) return data.insight_text;
    return "Analysis completed.";
  };

  const handleAnalyze = async (id) => {
    try {
      setAnalyzingId(id);
      setActionMessage("");
      const response = await analyzeScan(id);
      const updatedScan = response?.data;

      setActionMessage(
        `Scan ${id}: ${getPredictionLabel(updatedScan?.prediction)}. ${extractAnalyzeMessage(updatedScan)}`
      );
      await fetchScans();
    } catch (error) {
      console.error("Analyze failed", error?.response?.data || error.message);
      const backendMessage =
        error?.response?.data?.error ||
        error?.response?.data?.detail ||
        "Analyze failed.";
      setActionMessage(backendMessage);
    } finally {
      setAnalyzingId(null);
    }
  };

  useEffect(() => {
    fetchScans();
  }, []);

  return (
    <div>
      <h1 style={{ marginBottom: "12px" }}>All Scans</h1>
      <p style={{ marginBottom: "20px", color: "#94a3b8" }}>
        View uploaded scans, analysis status, confidence, and generated artifacts.
      </p>

      {actionMessage && (
        <div
          style={{
            marginBottom: "16px",
            padding: "12px 14px",
            borderRadius: "10px",
            background: "#0f172a",
            border: "1px solid #334155",
            color: "#e2e8f0",
          }}
        >
          {actionMessage}
        </div>
      )}

      {loading ? (
        <p>Loading scans...</p>
      ) : scans.length === 0 ? (
        <p>No scans found.</p>
      ) : (
        <div style={{ display: "grid", gap: "16px" }}>
          {scans.map((scan) => {
            const predictionColor = getPredictionColor(scan.prediction);
            const predictionLabel = getPredictionLabel(scan.prediction);

            return (
              <div
                key={scan.id}
                style={{
                  border: "1px solid #334155",
                  padding: "18px",
                  borderRadius: "14px",
                  background: "#0f172a",
                  color: "#fff",
                  boxShadow: "0 8px 24px rgba(0,0,0,0.18)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "flex-start",
                    gap: "12px",
                    flexWrap: "wrap",
                    marginBottom: "12px",
                  }}
                >
                  <div>
                    <h3 style={{ margin: 0, marginBottom: "6px" }}>Scan #{scan.id}</h3>
                    <p style={{ margin: 0, color: "#94a3b8" }}>
                      {scan.patient_name || "Unknown Patient"} • {scan.patient_id || "No Patient ID"}
                    </p>
                  </div>

                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      padding: "8px 12px",
                      borderRadius: "999px",
                      background: `${predictionColor}22`,
                      border: `1px solid ${predictionColor}`,
                      color: predictionColor,
                      fontWeight: 700,
                      fontSize: "14px",
                    }}
                  >
                    {predictionLabel}
                  </span>
                </div>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                    gap: "10px",
                    marginBottom: "14px",
                  }}
                >
                  <div>
                    <p style={{ margin: 0, color: "#94a3b8", fontSize: "14px" }}>Processed</p>
                    <p style={{ margin: 0 }}>{scan.is_processed ? "Yes" : "No"}</p>
                  </div>

                  <div>
                    <p style={{ margin: 0, color: "#94a3b8", fontSize: "14px" }}>Confidence</p>
                    <p style={{ margin: 0 }}>
                      {scan.confidence_score !== null && scan.confidence_score !== undefined
                        ? `${scan.confidence_score}%`
                        : "N/A"}
                    </p>
                  </div>

                  <div>
                    <p style={{ margin: 0, color: "#94a3b8", fontSize: "14px" }}>Uploaded</p>
                    <p style={{ margin: 0 }}>
                      {scan.uploaded_at
                        ? new Date(scan.uploaded_at).toLocaleString()
                        : "N/A"}
                    </p>
                  </div>
                </div>

                {scan.insight_text && (
                  <div
                    style={{
                      marginBottom: "12px",
                      padding: "12px",
                      borderRadius: "10px",
                      background: "#111827",
                      border: "1px solid #1f2937",
                    }}
                  >
                    <p style={{ margin: 0, color: "#cbd5e1" }}>
                      <strong>Insight:</strong> {scan.insight_text}
                    </p>
                  </div>
                )}

                {scan.validation_message && (
                  <div
                    style={{
                      marginBottom: "12px",
                      padding: "12px",
                      borderRadius: "10px",
                      background: "#3f1d0c",
                      border: "1px solid #9a3412",
                      color: "#fed7aa",
                    }}
                  >
                    <strong>Validation:</strong> {scan.validation_message}
                  </div>
                )}

                <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", marginTop: "12px" }}>
                  <button
                    onClick={() => handleAnalyze(scan.id)}
                    disabled={analyzingId === scan.id}
                    style={{
                      padding: "10px 14px",
                      borderRadius: "10px",
                      border: "none",
                      background: analyzingId === scan.id ? "#475569" : "#2563eb",
                      color: "#fff",
                      fontWeight: 700,
                      cursor: analyzingId === scan.id ? "not-allowed" : "pointer",
                    }}
                  >
                    {analyzingId === scan.id ? "Analyzing..." : "Analyze"}
                  </button>

                  {scan.report_file && (
                    <a
                      href={scan.report_file}
                      target="_blank"
                      rel="noreferrer"
                      style={{
                        padding: "10px 14px",
                        borderRadius: "10px",
                        background: "#16a34a",
                        color: "#fff",
                        textDecoration: "none",
                        fontWeight: 700,
                      }}
                    >
                      Open Report
                    </a>
                  )}

                  {scan.heatmap_image && (
                    <a
                      href={scan.heatmap_image}
                      target="_blank"
                      rel="noreferrer"
                      style={{
                        padding: "10px 14px",
                        borderRadius: "10px",
                        background: "#7c3aed",
                        color: "#fff",
                        textDecoration: "none",
                        fontWeight: 700,
                      }}
                    >
                      View Heatmap
                    </a>
                  )}

                  {scan.segmentation_mask && (
                    <a
                      href={scan.segmentation_mask}
                      target="_blank"
                      rel="noreferrer"
                      style={{
                        padding: "10px 14px",
                        borderRadius: "10px",
                        background: "#0f766e",
                        color: "#fff",
                        textDecoration: "none",
                        fontWeight: 700,
                      }}
                    >
                      View Mask
                    </a>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default Scans;