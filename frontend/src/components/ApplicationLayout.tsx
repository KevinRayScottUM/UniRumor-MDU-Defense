import { useEffect, useRef } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";

const navigationItems = [
  { label: "Home", href: "/", path: "/", hash: "" },
  { label: "Verify", href: "/#verify", path: "/", hash: "#verify" },
  { label: "Sessions", href: "/#sessions", path: "/", hash: "#sessions" },
  { label: "About MDU", href: "/about", path: "/about", hash: "" },
] as const;

const SITE_TITLE =
  "UniRumor-MDU: Explainable Multimodal Misinformation Verification";
const SITE_DESCRIPTION =
  "A research demonstration for claim-centered, evidence-unit-based multimodal misinformation verification.";

function routeMetadata(pathname: string) {
  if (pathname === "/about") {
    return {
      announcement: "About MDU",
      title: `About MDU | ${SITE_TITLE}`,
      description:
        "Learn how UniRumor-MDU organizes candidate evidence units and explanation references around one focal claim.",
    };
  }
  if (pathname === "/demo") {
    return {
      announcement: "Illustrative demo",
      title: `Illustrative Demo | ${SITE_TITLE}`,
      description:
        "Explore an explicitly illustrative UniRumor-MDU result layout without running a model or contacting the production API.",
    };
  }
  if (pathname.endsWith("/result")) {
    return {
      announcement: "Verification result",
      title: `Verification Result | ${SITE_TITLE}`,
      description: SITE_DESCRIPTION,
    };
  }
  if (pathname.startsWith("/jobs/")) {
    return {
      announcement: "Verification session",
      title: `Verification Session | ${SITE_TITLE}`,
      description: SITE_DESCRIPTION,
    };
  }
  return {
    announcement: "Home",
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
  };
}

function NavigationLinks({
  mobile = false,
  onNavigate,
}: {
  mobile?: boolean;
  onNavigate?: () => void;
}) {
  const location = useLocation();

  return (
    <nav aria-label={mobile ? "Mobile navigation" : "Primary navigation"}>
      <ul className={mobile ? "mobile-nav-list" : "primary-nav-list"}>
        {navigationItems.map((item) => {
          const active =
            location.pathname === item.path && location.hash === item.hash;
          return (
            <li key={item.label}>
              <a
                aria-current={active ? "page" : undefined}
                className={`navigation-link${active ? " navigation-link--active" : ""}`}
                href={item.href}
                onClick={onNavigate}
              >
                {item.label}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

export function ApplicationLayout() {
  const location = useLocation();
  const mainContent = useRef<HTMLElement>(null);
  const mobileNavigation = useRef<HTMLDetailsElement>(null);
  const previousPathname = useRef(location.pathname);
  const metadata = routeMetadata(location.pathname);

  useEffect(() => {
    document.title = metadata.title;
    document
      .querySelector<HTMLMetaElement>('meta[name="description"]')
      ?.setAttribute("content", metadata.description);
    document
      .querySelector<HTMLMetaElement>('meta[property="og:title"]')
      ?.setAttribute("content", metadata.title);
    document
      .querySelector<HTMLMetaElement>('meta[property="og:description"]')
      ?.setAttribute("content", metadata.description);
    document
      .querySelector<HTMLMetaElement>('meta[name="twitter:title"]')
      ?.setAttribute("content", metadata.title);
    document
      .querySelector<HTMLMetaElement>('meta[name="twitter:description"]')
      ?.setAttribute("content", metadata.description);
  }, [metadata.description, metadata.title]);

  useEffect(() => {
    if (previousPathname.current !== location.pathname) {
      mainContent.current?.focus({ preventScroll: true });
      previousPathname.current = location.pathname;
    }
  }, [location.pathname]);

  function closeMobileNavigation() {
    mobileNavigation.current?.removeAttribute("open");
  }

  return (
    <div className="application-shell" id="top">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <header className="site-header">
        <div className="site-header__inner">
          <Link
            aria-label="UniRumor MDU Defense home"
            className="brand"
            to="/"
          >
            <span aria-hidden="true" className="brand__mark">
              UM
            </span>
            <span className="brand__wordmark">
              <span className="brand__name">UniRumor-MDU</span>
              <span className="brand__descriptor">Research verification</span>
            </span>
          </Link>

          <div className="desktop-navigation">
            <NavigationLinks />
          </div>

          <div className="site-header__meta">
            <span aria-hidden="true" className="site-header__status-dot" />
            Research demo
          </div>

          <details className="mobile-navigation" ref={mobileNavigation}>
            <summary>Menu</summary>
            <div className="mobile-navigation__panel">
              <NavigationLinks mobile onNavigate={closeMobileNavigation} />
            </div>
          </details>
        </div>
      </header>

      <p aria-live="polite" className="visually-hidden">
        {metadata.announcement} page loaded
      </p>

      <main
        aria-label="Primary content"
        className="application-main"
        id="main-content"
        ref={mainContent}
        tabIndex={-1}
      >
        <div className="route-transition" key={location.pathname}>
          <Outlet />
        </div>
      </main>

      <footer className="site-footer">
        <div className="site-footer__inner">
          <div>
            <p className="site-footer__brand">UniRumor-MDU</p>
            <p className="site-footer__copy">
              Evidence-led multimodal verification for controlled research
              demonstration.
            </p>
          </div>
          <p className="site-footer__boundary">
            Scientific results remain authoritative to the production API.
          </p>
          <nav aria-label="Research resources" className="site-footer__links">
            <Link className="site-footer__method-link" to="/about">
              About the MDU method
              <span aria-hidden="true">→</span>
            </Link>
            <Link className="site-footer__method-link" to="/demo">
              Illustrative demo
              <span aria-hidden="true">→</span>
            </Link>
          </nav>
        </div>
      </footer>
    </div>
  );
}
