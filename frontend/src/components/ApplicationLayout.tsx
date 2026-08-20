import { NavLink, Outlet } from "react-router-dom";

export function ApplicationLayout() {
  return (
    <div className="application-shell">
      <header className="application-header">
        <NavLink className="application-title" to="/">
          UniRumor MDU Defense
        </NavLink>
        <p className="application-subtitle">Production verification interface</p>
      </header>
      <main className="application-main">
        <Outlet />
      </main>
      <footer className="application-footer">
        Results are provided by the authoritative production API.
      </footer>
    </div>
  );
}

