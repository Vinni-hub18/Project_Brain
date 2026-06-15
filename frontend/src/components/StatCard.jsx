function StatCard({ title, value, subtitle, color = "#60a5fa" }) {
  return (
    <div className="stat-card">
      <p className="stat-title">{title}</p>
      <h3 className="stat-value" style={{ color }}>
        {value}
      </h3>
      <p className="stat-subtitle">{subtitle}</p>
    </div>
  );
}

export default StatCard;