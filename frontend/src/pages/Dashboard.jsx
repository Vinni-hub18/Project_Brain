import { useEffect, useMemo, useState } from "react";
import {
  getAllScans,
  analyzeAllScans,
  getPredictionColor,
  getPredictionLabel,
} from "../services/scanService";

function Dashboard() {
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [runningBatch, setRunningBatch] = useState(false);
  const [message, setMessage] = useState("");

  const fetchScans = async () => {
    try {
      setLoading(true);
      setMessage("");
      const response = await getAllScans();
      setScans(response.data || []);
    } catch (error) {
      console.error("Failed to fetch dashboard data:", error?.response?.data || error.message);
      setMessage("Failed to load dashboard data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchScans();
  }, []);

  const stats = useMemo(() => {
    const total = scans.length;
    const processed = scans.filter((scan) => scan.is_processed).length;
    const pending = scans.filter((scan) => !scan.is_processed || scan.prediction === "pending").length;
    const tumorSuspected = scans.filter(
      (scan) => scan.prediction === "tumor_suspected" || scan.prediction === "tumor"
    ).length;
    const noTumor = scans.filter((scan) => scan.prediction === "no_tumor").length;
    const uncertain = scans.filter((scan) => scan.prediction === "uncertain").length;
    const invalidInput = scans.filter((scan) => scan.prediction === "invalid_input").length;

    return {
      total,
      processed,
      pending,
      tumorSuspected,
      noTumor,
      uncertain,
      invalidInput,
    };
  }, [scans]);

  const recentScans = useMemo(() => {
    return [...scans]
      .sort((a, b) => new Date(b.uploaded_at || 0) - new Date(a.uploaded_at || 0))
      .slice(0, 5);
  }, [scans]);

  const handleAnalyzeAll = async () => {
    try {
      setRunningBatch(true);
      setMessage("");
      const response = await analyzeAllScans();
      const resultCount = response?.data?.results?.length || 0;
      setMessage(
        resultCount > 0
          ? `Batch analysis complete. ${resultCount} scan(s) processed.`
          : response?.data?.message || "Batch analysis complete."
      );
      await fetchScans();
    } catch (error) {
      console.error("Batch analyze failed:", error?.response?.data || error.message);
      setMessage(
        error?.response?.data?.message ||
          error?.response?.data?.error ||
          "Batch analysis failed."
      );
    } finally {
      setRunningBatch(false);
    }
  };

  const StatCard = ({ title, value, subtitle, color = "#2563eb" }) => (
    <div
      style={{
        background: "#ffffff",
        border: "1px solid #dbe4ee",
        borderRadius: "16px",
        padding: "18px",
        boxShadow: "0 10px 30px rgba(15, 23, 42, 0.06)",
      }}
    >
      <p
        style={{
          margin: 0,
          marginBottom: "8px",
          color: "#64748b",
          fontSize: "14px",
          fontWeight: 600,
        }}
      >
        {title}
      </p>
      <h2
        style={{
          margin: 0,
          fontSize: "32px",
          lineHeight: 1.1,
          color,
        }}
      >
        {value}
      </h2>
      <p
        style={{
          margin: 0,
          marginTop: "8px",
          color: "#475569",
          fontSize: "14px",
        }}
      >
        {subtitle}
      </p>
    </div>
  );

  return (
    <div style={{ color: "#0f172a" }}>
      <div
        style={{
          marginBottom: "24px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: "16px",
          flexWrap: "wrap",
        }}
      >
        <div>
          <h1 style={{ margin: 0, marginBottom: "10px", fontSize: "40px", color: "#0f172a" }}>
            Dashboard
          </h1>
          <p style={{ margin: 0, color: "#64748b", maxWidth: "700px" }}>
            Monitor uploads, analysis progress, and prediction outcomes for your brain tumor
            detection workflow.
          </p>
        </div>

        <button
          onClick={handleAnalyzeAll}
          disabled={runningBatch}
          style={{
            padding: "12px 16px",
            borderRadius: "12px",
            border: "none",
            background: runningBatch ? "#94a3b8" : "#2563eb",
            color: "#ffffff",
            fontWeight: 700,
            cursor: runningBatch ? "not-allowed" : "pointer",
            minWidth: "170px",
            boxShadow: runningBatch ? "none" : "0 10px 20px rgba(37, 99, 235, 0.18)",
          }}
        >
          {runningBatch ? "Analyzing..." : "Analyze All Pending"}
        </button>
      </div>

      {message && (
        <div
          style={{
            marginBottom: "18px",
            padding: "12px 14px",
            borderRadius: "12px",
            background: "#eff6ff",
            border: "1px solid #bfdbfe",
            color: "#1d4ed8",
          }}
        >
          {message}
        </div>
      )}

      {loading ? (
        <p style={{ color: "#475569" }}>Loading dashboard...</p>
      ) : (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
              gap: "16px",
              marginBottom: "24px",
            }}
          >
            <StatCard
              title="Total Scans"
              value={stats.total}
              subtitle="All uploaded scan records"
              color="#2563eb"
            />
            <StatCard
              title="Processed"
              value={stats.processed}
              subtitle="Scans already analyzed"
              color="#16a34a"
            />
            <StatCard
              title="Pending"
              value={stats.pending}
              subtitle="Awaiting analysis"
              color="#d97706"
            />
            <StatCard
              title="Tumor Suspected"
              value={stats.tumorSuspected}
              subtitle="Needs careful review"
              color="#dc2626"
            />
            <StatCard
              title="No Tumor"
              value={stats.noTumor}
              subtitle="No significant region detected"
              color="#059669"
            />
            <StatCard
              title="Uncertain"
              value={stats.uncertain}
              subtitle="Low-confidence suspicious region"
              color="#f97316"
            />
            <StatCard
              title="Invalid Input"
              value={stats.invalidInput}
              subtitle="Rejected by validation checks"
              color="#7c3aed"
            />
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)",
              gap: "18px",
            }}
          >
            <div
              style={{
                background: "#ffffff",
                border: "1px solid #dbe4ee",
                borderRadius: "16px",
                padding: "18px",
                boxShadow: "0 10px 30px rgba(15, 23, 42, 0.06)",
              }}
            >
              <h2 style={{ marginTop: 0, marginBottom: "14px", fontSize: "22px", color: "#0f172a" }}>
                Recent Scans
              </h2>

              {recentScans.length === 0 ? (
                <p style={{ color: "#64748b" }}>No scans uploaded yet.</p>
              ) : (
                <div style={{ display: "grid", gap: "12px" }}>
                  {recentScans.map((scan) => {
                    const badgeColor = getPredictionColor(scan.prediction);
                    const label = getPredictionLabel(scan.prediction);

                    return (
                      <div
                        key={scan.id}
                        style={{
                          padding: "14px",
                          borderRadius: "12px",
                          background: "#f8fafc",
                          border: "1px solid #e2e8f0",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "flex-start",
                            gap: "12px",
                            flexWrap: "wrap",
                            marginBottom: "8px",
                          }}
                        >
                          <div>
                            <p style={{ margin: 0, fontWeight: 700, color: "#0f172a" }}>
                              Scan #{scan.id}
                            </p>
                            <p style={{ margin: 0, color: "#64748b", fontSize: "14px" }}>
                              {scan.patient_name || "Unknown Patient"} •{" "}
                              {scan.patient_id || "No Patient ID"}
                            </p>
                          </div>

                          <span
                            style={{
                              padding: "6px 10px",
                              borderRadius: "999px",
                              background: `${badgeColor}14`,
                              border: `1px solid ${badgeColor}`,
                              color: badgeColor,
                              fontWeight: 700,
                              fontSize: "13px",
                            }}
                          >
                            {label}
                          </span>
                        </div>

                        <div
                          style={{
                            display: "grid",
                            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
                            gap: "10px",
                          }}
                        >
                          <div>
                            <p style={{ margin: 0, color: "#64748b", fontSize: "13px" }}>
                              Confidence
                            </p>
                            <p style={{ margin: 0, color: "#0f172a" }}>
                              {scan.confidence_score !== null &&
                              scan.confidence_score !== undefined
                                ? `${scan.confidence_score}%`
                                : "N/A"}
                            </p>
                          </div>
                          <div>
                            <p style={{ margin: 0, color: "#64748b", fontSize: "13px" }}>
                              Processed
                            </p>
                            <p style={{ margin: 0, color: "#0f172a" }}>
                              {scan.is_processed ? "Yes" : "No"}
                            </p>
                          </div>
                          <div>
                            <p style={{ margin: 0, color: "#64748b", fontSize: "13px" }}>
                              Uploaded
                            </p>
                            <p style={{ margin: 0, color: "#0f172a" }}>
                              {scan.uploaded_at
                                ? new Date(scan.uploaded_at).toLocaleString()
                                : "N/A"}
                            </p>
                          </div>
                        </div>

                        {scan.insight_text && (
                          <p style={{ marginTop: "10px", marginBottom: 0, color: "#475569" }}>
                            <strong style={{ color: "#0f172a" }}>Insight:</strong>{" "}
                            {scan.insight_text}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            <div
              style={{
                background: "#ffffff",
                border: "1px solid #dbe4ee",
                borderRadius: "16px",
                padding: "18px",
                boxShadow: "0 10px 30px rgba(15, 23, 42, 0.06)",
              }}
            >
              <h2 style={{ marginTop: 0, marginBottom: "14px", fontSize: "22px", color: "#0f172a" }}>
                System Summary
              </h2>

              <div style={{ display: "grid", gap: "12px" }}>
                <div
                  style={{
                    padding: "14px",
                    borderRadius: "12px",
                    background: "#f8fafc",
                    border: "1px solid #e2e8f0",
                  }}
                >
                  <p style={{ margin: 0, color: "#64748b", fontSize: "14px" }}>
                    Workflow Status
                  </p>
                  <p style={{ margin: 0, marginTop: "6px", fontWeight: 700, color: "#0f172a" }}>
                    {stats.pending > 0
                      ? `${stats.pending} scan(s) waiting for analysis`
                      : "All scans are currently processed"}
                  </p>
                </div>

                <div
                  style={{
                    padding: "14px",
                    borderRadius: "12px",
                    background: "#f8fafc",
                    border: "1px solid #e2e8f0",
                  }}
                >
                  <p style={{ margin: 0, color: "#64748b", fontSize: "14px" }}>
                    Highest Priority
                  </p>
                  <p style={{ margin: 0, marginTop: "6px", fontWeight: 700, color: "#0f172a" }}>
                    {stats.tumorSuspected > 0
                      ? `${stats.tumorSuspected} tumor-suspected case(s)`
                      : "No tumor-suspected cases right now"}
                  </p>
                </div>

                <div
                  style={{
                    padding: "14px",
                    borderRadius: "12px",
                    background: "#f8fafc",
                    border: "1px solid #e2e8f0",
                  }}
                >
                  <p style={{ margin: 0, color: "#64748b", fontSize: "14px" }}>
                    Input Quality
                  </p>
                  <p style={{ margin: 0, marginTop: "6px", fontWeight: 700, color: "#0f172a" }}>
                    {stats.invalidInput > 0
                      ? `${stats.invalidInput} invalid input scan(s) detected`
                      : "No invalid inputs detected"}
                  </p>
                </div>

                <div
                  style={{
                    padding: "14px",
                    borderRadius: "12px",
                    background: "#f8fafc",
                    border: "1px solid #e2e8f0",
                  }}
                >
                  <p style={{ margin: 0, color: "#64748b", fontSize: "14px" }}>
                    Last Refreshed
                  </p>
                  <p style={{ margin: 0, marginTop: "6px", fontWeight: 700, color: "#0f172a" }}>
                    {new Date().toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default Dashboard;