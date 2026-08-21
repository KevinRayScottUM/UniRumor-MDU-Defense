# Cloudflare Pages deployment

This frontend is a static React and Vite application. It communicates only
with the public FastAPI contract; no scientific runtime or server filesystem is
part of the Pages deployment.

## Pages build configuration

Configure the Pages project with these values:

| Setting | Value |
| --- | --- |
| Root directory | `frontend` |
| Build command | `npm run build` |
| Build output directory | `dist` |
| Node.js version | `22.16.0` (from `.node-version`) |

Cloudflare Pages installs the locked npm dependencies before running the build.
The Vite configuration emits root-relative, hashed assets into `dist` and
empties stale output before every build.

## Public API environment

`VITE_API_BASE_URL` is the only frontend deployment variable.

- Leave it unset or empty when Pages and the API are exposed through the same
  origin.
- Set it to the public API origin, or an HTTP(S) gateway prefix, when the API is
  hosted separately. Do not add a trailing slash.
- Configure preview and production values separately under the Pages project's
  **Settings > Environment variables**.
- Rebuild the frontend after changing the value. Vite substitutes it at build
  time.
- Never place credentials, API keys, internal paths, or other secrets in a
  `VITE_` variable. Vite includes these values in the public browser bundle.

The frontend rejects non-HTTP(S), credential-bearing, query-bearing, and
fragment-bearing base URLs with fixed public-safe configuration text. Empty
configuration remains a same-origin default and no deployment domain is
embedded in source code.

For cross-origin deployments, the API deployment must allow the exact Pages
preview and production origins through its existing CORS configuration. This is
an environment prerequisite, not a frontend API-contract change.

## SPA routing and static assets

The application uses browser routes such as `/about`, `/demo`,
`/jobs/:jobId`, and `/jobs/:jobId/result`. The build intentionally has no
top-level `404.html`; Cloudflare Pages therefore applies its documented SPA
behavior and serves the root application for unmatched routes. Do not add a
top-level `404.html` without also designing an equivalent route fallback.

The Vite base path is `/`, so built JavaScript, CSS, favicon, Apple touch icon,
and Open Graph image resolve from the Pages site root even after a nested-route
refresh.

## Local production check

```sh
npm ci
npm run check
npm test
npm run build
npm run preview
```

Before promoting a deployment:

1. Open `/`, `/about`, and `/demo` directly.
2. Refresh each nested route and confirm the application shell still renders.
3. Open representative job and result URLs and confirm failures use only the
   backend public error envelope or the frontend's fixed safe fallback text.
4. Confirm the generated JavaScript and CSS, `/favicon.png`, and `/og.png`
   return successfully.
5. Confirm the configured public API responds at `/api/v1/health` and
   `/api/v1/readiness` from the deployed browser origin.

No deployment is performed by this repository change.
