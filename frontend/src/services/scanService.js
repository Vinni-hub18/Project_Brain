import api from "./api";

export const getAllScans = () => api.get("/scans/");
export const getPendingScans = () => api.get("/scans/pending/");
export const getScanById = (id) => api.get(`/scans/${id}/`);
export const analyzeScan = (id) => api.post(`/scans/${id}/analyze/`);
export const analyzeAllScans = () => api.post("/scans/analyze-all/");
export const sendChatMessage = (message) => api.post("/chat/", { message });

export const uploadScan = (formData) =>
  api.post("/scans/upload/", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

export const uploadMultipleScans = (formData) =>
  api.post("/scans/upload-multiple/", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

export const getPredictionLabel = (prediction) => {
  const labels = {
    pending: "Pending Analysis",
    tumor: "Tumor",
    no_tumor: "No Tumor",
    tumor_suspected: "Tumor Suspected",
    uncertain: "Uncertain",
    invalid_input: "Invalid Input",
  };

  return labels[prediction] || prediction || "Unknown";
};

export const getPredictionColor = (prediction) => {
  const colors = {
    pending: "#f59e0b",
    tumor: "#ef4444",
    no_tumor: "#22c55e",
    tumor_suspected: "#ef4444",
    uncertain: "#f97316",
    invalid_input: "#a855f7",
  };

  return colors[prediction] || "#94a3b8";
};