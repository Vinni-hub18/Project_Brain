export const formatDateTime = (value) => {
  if (!value) return "N/A";

  try {
    return new Date(value).toLocaleString("en-IN", {
      day: "numeric",
      month: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "N/A";
  }
};

export const formatConfidence = (value) => {
  if (value === null || value === undefined || value === "") return "N/A";
  const num = Number(value);
  if (Number.isNaN(num)) return "N/A";
  return `${num.toFixed(2)}%`;
};

export const normalizePrediction = (prediction) => {
  if (!prediction) return "unknown";
  return String(prediction).trim().toLowerCase();
};

export const getPredictionLabel = (prediction) => {
  const key = normalizePrediction(prediction);

  const labels = {
    pending: "Pending Analysis",
    tumor: "Tumor",
    tumor_suspected: "Tumor Suspected",
    no_tumor: "No Tumor",
    uncertain: "Uncertain",
    invalid_input: "Invalid Input",
    unknown: "Unknown",
  };

  return labels[key] || prediction;
};

export const getPredictionColor = (prediction) => {
  const key = normalizePrediction(prediction);

  const colors = {
    pending: "#f59e0b",
    tumor: "#ef4444",
    tumor_suspected: "#ef4444",
    no_tumor: "#22c55e",
    uncertain: "#f97316",
    invalid_input: "#a855f7",
    unknown: "#94a3b8",
  };

  return colors[key] || "#94a3b8";
};

export const getStatusTone = (prediction) => {
  const color = getPredictionColor(prediction);
  return {
    color,
    background: `${color}22`,
    border: `1px solid ${color}`,
  };
};

export const countStats = (scans = []) => {
  return scans.reduce(
    (acc, scan) => {
      const prediction = normalizePrediction(scan.prediction);

      acc.total += 1;
      if (scan.is_processed) acc.processed += 1;
      if (!scan.is_processed || prediction === "pending") acc.pending += 1;
      if (prediction === "tumor" || prediction === "tumor_suspected") acc.tumor_suspected += 1;
      if (prediction === "no_tumor") acc.no_tumor += 1;
      if (prediction === "uncertain") acc.uncertain += 1;
      if (prediction === "invalid_input") acc.invalid_input += 1;

      return acc;
    },
    {
      total: 0,
      processed: 0,
      pending: 0,
      tumor_suspected: 0,
      no_tumor: 0,
      uncertain: 0,
      invalid_input: 0,
    }
  );
};

export const extractApiErrorMessage = (error, fallback = "Something went wrong.") => {
  const data = error?.response?.data;

  if (!data) return fallback;
  if (typeof data === "string") return data;
  if (data.error) return data.error;
  if (data.message) return data.message;
  if (data.detail) return data.detail;

  const firstKey = Object.keys(data)[0];
  if (firstKey) {
    const value = data[firstKey];
    if (Array.isArray(value) && value.length > 0) {
      return `${firstKey}: ${value[0]}`;
    }
    if (typeof value === "string") {
      return `${firstKey}: ${value}`;
    }
  }

  return fallback;
};

export const getGreeting = () => {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
};