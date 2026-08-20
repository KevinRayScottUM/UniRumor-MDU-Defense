# UniRumor MDU Defense frontend

This directory contains the independent React, TypeScript, and Vite frontend.
It consumes only the public HTTP API and contains no production runtime or
scientific inference logic.

Set `VITE_API_BASE_URL` to the public API origin for a deployment. Leave it
unset for same-origin development or reverse-proxy setups.

```sh
npm install
npm run check
npm test
npm run build
```

