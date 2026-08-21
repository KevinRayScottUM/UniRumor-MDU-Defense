# UniRumor MDU Defense frontend

This directory contains the independent React, TypeScript, and Vite frontend.
It consumes only the public HTTP API and contains no production runtime or
scientific inference logic.

## Local development

Use Node.js 22.16 or newer. From this directory:

```sh
npm install
npm run dev
```

Vite prints the local preview URL. Stop the process with `Ctrl+C` when finished.

The application uses same-origin API requests by default. Copy `.env.example`
to `.env.local` and set `VITE_API_BASE_URL` only when the frontend and public
API use different origins. This build-time variable must contain an absolute
HTTP(S) URL and must never contain credentials or secrets.

## Quality and production build

```sh
npm run check
npm test
npm run build
```

Run `npm run preview` after a production build to inspect the generated site.
Cloudflare Pages settings and the release checklist are documented in
[`DEPLOYMENT.md`](./DEPLOYMENT.md).
