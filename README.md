# UniRumor MDU — Defense Engineering System

This branch contains the engineering-oriented graduation defense system.

Target runtime pipeline:

Video + Claim
-> video/audio preprocessing
-> ASR
-> OCR
-> CLIP/SigLIP frame retrieval
-> VLM visual observations
-> MDU candidate construction
-> Frozen G1 DeBERTa inference
-> sample-level verdict
-> Top-k explanatory units
-> Web application visualization

The frozen scientific model is used as a read-only decision engine.
Engineering development must not modify the frozen official Test results.

## Web frontend

The public research-demo frontend is an independent React, TypeScript, and Vite
application under [`frontend/`](./frontend/). It communicates only through the
public web API and does not run models or reconstruct scientific outcomes.

Use Node.js 22.16 or newer, then start the development site with:

```sh
cd frontend
npm install
npm run dev
```

Create the production static bundle with:

```sh
npm run build
```

See [`frontend/README.md`](./frontend/README.md) for frontend development and
[`frontend/DEPLOYMENT.md`](./frontend/DEPLOYMENT.md) for Cloudflare Pages
configuration and release checks.
