import { Link } from "react-router-dom";
import {
  formatConfidence,
  formatDateTime,
  getPredictionLabel,
  getStatusTone,
} from "../utils/helpers";

function ScanCard({ scan, onAnalyze, analyzing = false }) {
  const tone = getStatusTone(scan.prediction);

  return (
    <div className="scan-card">
      <div className="scan-card-header">
        <div>
          <h3 className="scan-card-title">
            {scan.patient_name || "Unknown Patient"}
          </h3>
          <p className="scan-card-subtitle">
            Scan #{scan.id} • {scan.patient_id || "No Patient ID"}
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

      <div className="meta-grid">
        <div className="meta-item">
          <p className="meta-label">Confidence</p>
          <p className="meta-value">{formatConfidence(scan.confidence_score)}</p>
        </div>

        <div className="meta-item">
          <p className="meta-label">Processed</p>
          <p className="meta-value">{scan.is_processed ? "Yes" : "No"}</p>
        </div>

        <div className="meta-item">
          <p className="meta-label">Uploaded</p>
          <p className="meta-value">{formatDateTime(scan.uploaded_at)}</p>
        </div>

        <div className="meta-item">
          <p className="meta-label">Updated</p>
          <p className="meta-value">{formatDateTime(scan.updated_at)}</p>
        </div>
      </div>

      {scan.insight_text && (
        <div className="scan-note">
          <strong>Insight:</strong> {scan.insight_text}
        </div>
      )}

      <div className="scan-card-actions">
        <Link to={`/scans/${scan.id}`} className="btn btn-secondary">
          View Details
        </Link>

        {!scan.is_processed && (
          <button
            className="btn btn-primary"
            onClick={() => onAnalyze?.(scan.id)}
            disabled={analyzing}
          >
            {analyzing ? "Analyzing..." : "Analyze"}
          </button>
        )}

        {scan.report_file && (
          <a
            href={scan.report_file}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-success"
          >
            Open Report
          </a>
        )}
      </div>
    </div>
  );
}

export default ScanCard;