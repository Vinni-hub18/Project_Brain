import { useState } from "react";
import { uploadScan, uploadMultipleScans } from "../services/scanService";
import { extractApiErrorMessage } from "../utils/helpers";

function Upload() {
  const [patientName, setPatientName] = useState("");
  const [patientId, setPatientId] = useState("");
  const [singleFile, setSingleFile] = useState(null);
  const [multipleFiles, setMultipleFiles] = useState([]);
  const [message, setMessage] = useState("");
  const [uploadingSingle, setUploadingSingle] = useState(false);
  const [uploadingMultiple, setUploadingMultiple] = useState(false);

  const handleSingleUpload = async (e) => {
    e.preventDefault();

    if (!patientName || !patientId || !singleFile) {
      setMessage("Please fill patient details and select one scan file.");
      return;
    }

    try {
      setUploadingSingle(true);
      setMessage("");

      const formData = new FormData();
      formData.append("patient_name", patientName);
      formData.append("patient_id", patientId);
      formData.append("scan_file", singleFile);

      await uploadScan(formData);
      setMessage("Single scan uploaded successfully.");
      setSingleFile(null);
    } catch (error) {
      setMessage(extractApiErrorMessage(error, "Single upload failed."));
    } finally {
      setUploadingSingle(false);
    }
  };

  const handleMultipleUpload = async (e) => {
    e.preventDefault();

    if (!patientName || !patientId || multipleFiles.length === 0) {
      setMessage("Please fill patient details and select one or more scan files.");
      return;
    }

    try {
      setUploadingMultiple(true);
      setMessage("");

      const formData = new FormData();
      formData.append("patient_name", patientName);
      formData.append("patient_id", patientId);

      multipleFiles.forEach((file) => {
        formData.append("scan_file", file);
      });

      await uploadMultipleScans(formData);
      setMessage("Multiple scans uploaded successfully.");
      setMultipleFiles([]);
    } catch (error) {
      setMessage(extractApiErrorMessage(error, "Multiple upload failed."));
    } finally {
      setUploadingMultiple(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Upload Scans</h1>
          <p className="page-subtitle">
            Upload a single scan or batch upload multiple images for processing.
          </p>
        </div>
      </div>

      {message && <div className="message-box">{message}</div>}

      <div className="upload-grid">
        <div className="upload-card">
          <h2 className="panel-title">Single Upload</h2>
          <p className="panel-subtitle">Upload one MRI scan at a time.</p>

          <form onSubmit={handleSingleUpload} className="form-grid">
            <div className="form-group">
              <label className="form-label">Patient Name</label>
              <input
                type="text"
                className="form-input"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                placeholder="Enter patient name"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Patient ID</label>
              <input
                type="text"
                className="form-input"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                placeholder="Enter patient ID"
              />
            </div>

            <div className="form-group form-group-full">
              <label className="form-label">Scan File</label>
              <input
                type="file"
                className="form-file"
                accept=".jpg,.jpeg,.png,.dcm"
                onChange={(e) => setSingleFile(e.target.files[0] || null)}
              />
            </div>

            <div className="form-group form-group-full">
              <button type="submit" className="btn btn-primary" disabled={uploadingSingle}>
                {uploadingSingle ? "Uploading..." : "Upload Single Scan"}
              </button>
            </div>
          </form>
        </div>

        <div className="upload-card">
          <h2 className="panel-title">Multiple Upload</h2>
          <p className="panel-subtitle">Select several images and send them in one request.</p>

          <form onSubmit={handleMultipleUpload} className="form-grid">
            <div className="form-group form-group-full">
              <label className="form-label">Select Multiple Files</label>
              <input
                type="file"
                className="form-file"
                accept=".jpg,.jpeg,.png,.dcm"
                multiple
                onChange={(e) => setMultipleFiles(Array.from(e.target.files || []))}
              />
            </div>

            <div className="form-group form-group-full">
              <div className="scan-note">
                <strong>Selected Files:</strong>{" "}
                {multipleFiles.length
                  ? multipleFiles.map((file) => file.name).join(", ")
                  : "No files selected"}
              </div>
            </div>

            <div className="form-group form-group-full">
              <button type="submit" className="btn btn-primary" disabled={uploadingMultiple}>
                {uploadingMultiple ? "Uploading..." : "Upload Multiple Scans"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default Upload;