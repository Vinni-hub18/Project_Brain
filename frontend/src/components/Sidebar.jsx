import { NavLink } from "react-router-dom";
import logo from "./logo3.png"; // make sure this file exists

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-copy">
          <img src={logo} alt="NuroScan Logo" className="logo-image" />
          <div>
            
            <p>Brain MRI analysis workspace</p>
          </div>
        </div>
      </div>

      <div>
        <p className="sidebar-group-title">Workspace</p>

        <nav className="sidebar-nav">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `sidebar-link ${isActive ? "active" : ""}`
            }
          >
            <span>Dashboard</span>
          </NavLink>

          <NavLink
            to="/upload"
            className={({ isActive }) =>
              `sidebar-link ${isActive ? "active" : ""}`
            }
          >
            <span>Upload</span>
          </NavLink>

          <NavLink
            to="/scans"
            className={({ isActive }) =>
              `sidebar-link ${isActive ? "active" : ""}`
            }
          >
            <span>All Scans</span>
          </NavLink>

          <NavLink
            to="/pending"
            className={({ isActive }) =>
              `sidebar-link ${isActive ? "active" : ""}`
            }
          >
            <span>Pending</span>
          </NavLink>

          <NavLink
            to="/chatbot"
            className={({ isActive }) =>
              `sidebar-link ${isActive ? "active" : ""}`
            }
          >
            <span>Assistant</span>
          </NavLink>
        </nav>
      </div>

      <div className="sidebar-footer">
        <p>
          AI-assisted diagnostic support. Final diagnosis must always be
          confirmed by a qualified doctor.
        </p>
      </div>
    </aside>
  );
}

export default Sidebar;