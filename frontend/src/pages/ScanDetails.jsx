import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import Loader from "../components/Loader";
import { analyzeScan, getScanById } from "../services/scanService";
import {
  extractApiErrorMessage,
  formatConfidence,
  formatDateTime,
  getPredictionLabel,
  getStatusTone,
} from "../utils/helpers";

function ScanDetails() {
  const { id } = useParams();
  const [scan, setScan] = useState(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [message, setMessage] = useState("");

  const fetchScan = async () => {
    try {
      setLoading(true);
      setMessage("");
      const response = await getScanById(id);
      setScan(response.data);
    } catch (error) {
      console.error("Failed to fetch scan details:", error?.response?.data || error.message);
      setMessage(extractApiErrorMessage(error, "Failed to fetch scan details."));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (id) {
      fetchScan();
    }
  }, [id]);

  const handleAnalyze = async () => {
    try {
      setAnalyzing(true);
      setMessage("");
      const response = await analyzeScan(id);
      setScan(response.data);
      setMessage("Scan analyzed successfully.");
    } catch (error) {
      console.error("Analyze failed:", error?.response?.data || error.message);
      setMessage(extractApiErrorMessage(error, "Analyze failed."));
    } finally {
      setAnalyzing(false);
    }
  };

  if (loading) {
    return <Loader text="Loading scan details..." />;
  }

  if (!scan) {
    return (
      <div className="empty-panel">
        <h1>Scan not found</h1>
        <p className="section-text">The requested scan could not be loaded.</p>
        <Link to="/scans" className="btn btn-secondary">
          Back to Scans
        </Link>
      </div>
    );
  }

  const tone = getStatusTone(scan.prediction);

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Scan Details</h1>
          <p className="section-text">
            Detailed prediction, scan metadata, generated report, and visual outputs.
          </p>
        </div>

        <div className="page-header-actions">
          <Link to="/scans" className="btn btn-secondary">
            Back to Scans
          </Link>
          <button
            className="btn btn-primary"
            onClick={handleAnalyze}
            disabled={analyzing}
          >
            {analyzing ? "Analyzing..." : "Analyze Again"}
          </button>
        </div>
      </div>

      {message && <div className="message-box">{message}</div>}

      <div className="details-layout">
        <section className="panel">
          <div className="scan-card-header">
            <div>
              <h2 className="panel-title">Scan #{scan.id}</h2>
              <p className="section-text">
                {scan.patient_name || "Unknown Patient"} • {scan.patient_id || "No Patient ID"}
              </p>
            </div>

            <span
              className="status-badge"
              style={{
                color: tone.color,
                background: tone.background,
                border: tone.border,
              }}
            >
              {getPredictionLabel(scan.prediction)}
            </span>
          </div>

          <div className="detail-grid">
            <div className="detail-item">
              <p className="meta-label">Confidence</p>
              <p className="meta-value">{formatConfidence(scan.confidence_score)}</p>
            </div>

            <div className="detail-item">
              <p className="meta-label">Processed</p>
              <p className="meta-value">{scan.is_processed ? "Yes" : "No"}</p>
            </div>

            <div className="detail-item">
              <p className="meta-label">Uploaded At</p>
              <p className="meta-value">{formatDateTime(scan.uploaded_at)}</p>
            </div>

            <div className="detail-item">
              <p className="meta-label">Updated At</p>
              <p className="meta-value">{formatDateTime(scan.updated_at)}</p>
            </div>
          </div>

          {scan.insight_text && (
            <div className="scan-note">
              <strong>Insight:</strong> {scan.insight_text}
            </div>
          )}

          {scan.validation_message && (
            <div className="scan-warning">
              <strong>Validation:</strong> {scan.validation_message}
            </div>
          )}

          <div className="scan-card-actions">
            {scan.report_file && (
              <a
                className="btn btn-success"
                href={scan.report_file}
                target="_blank"
                rel="noopener noreferrer"
              >
                Open Report
              </a>
            )}

            {scan.scan_file && (
              <a
                className="btn btn-secondary"
                href={scan.scan_file}
                target="_blank"
                rel="noopener noreferrer"
              >
                Original Scan
              </a>
            )}
          </div>
        </section>

        <section className="panel">
          <h2 className="panel-title">Generated Outputs</h2>

          <div className="media-grid">
            <div className="media-card">
              <h3 className="media-title">Heatmap</h3>
              {scan.heatmap_image ? (
                <>
                  <img
                    src={scan.heatmap_image}
                    alt="Heatmap output"
                    className="scan-image"
                  />
                  <a
                    href={scan.heatmap_image}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-secondary"
                  >
                    Open Heatmap
                  </a>
                </>
              ) : (
                <p className="section-text">No heatmap available yet.</p>
              )}
            </div>

            <div className="media-card">
              <h3 className="media-title">Segmentation Mask</h3>
              {scan.segmentation_mask ? (
                <>
                  <img
                    src={scan.segmentation_mask}
                    alt="Segmentation mask output"
                    className="scan-image"
                  />
                  <a
                    href={scan.segmentation_mask}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-secondary"
                  >
                    Open Mask
                  </a>
                </>
              ) : (
                <p className="section-text">No segmentation mask available yet.</p>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default ScanDetails;