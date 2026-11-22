# Docker: local read-only research viewer

## Production image

The build stage uses Node 24 LTS, `npm ci` and `npm run build`. The runtime stage contains nginx Alpine and `dist` only. Python, the exporter, reference repositories and private snapshots are not runtime dependencies. `.dockerignore` excludes environment files, runtime snapshots, data, weights, reports and caches; explicit COPY instructions further restrict the build inputs.

nginx serves SPA routes with `try_files $uri $uri/ /index.html`. Missing `/runtime/*` and `/assets/*` return 404 instead of HTML. Hashed assets are immutable; the SPA is revalidated; runtime config and mutable snapshots use `no-store`. JSON/JS/CSS responses support gzip. `/health` returns `healthy`; the Dockerfile healthcheck probes it. Compose binds only localhost port 8080.

## DEMO

```powershell
docker build -t flir-pipeline-explorer .
docker compose up -d --build --wait
docker compose ps
curl.exe --fail http://localhost:8080/health
curl.exe --fail http://localhost:8080/organization/sequences
docker compose down
```

Without overrides, the viewer starts in DEMO. Open `http://localhost:8080`.

## REAL / MIXED

Place the existing sanitized snapshot at ignored `public/runtime/leakage-snapshot.json`. Compose mounts the directory at `/usr/share/nginx/html/runtime:ro`; the snapshot is never copied into the image.

```powershell
$env:FLIR_DATA_MODE = 'real'
$env:FLIR_LEAKAGE_SNAPSHOT_URL = '/runtime/leakage-snapshot.json'
docker compose up -d --force-recreate --wait
```

POSIX equivalent:

```sh
FLIR_DATA_MODE=real FLIR_LEAKAGE_SNAPSHOT_URL=/runtime/leakage-snapshot.json docker compose up -d --force-recreate --wait
```

Changing these variables requires recreating the container, **not rebuilding the image**. Replacing an existing mounted snapshot requires reloading the browser; the adapter requests it with `no-store`. Query caches the validated snapshot for the current browser session. M01/M03 remain DEMO. An invalid/missing real snapshot produces a recoverable error, never a silent scientific fallback.

## Runtime configuration

The nginx entrypoint writes `window.__FLIR_CONFIG__` to `/runtime-config.js` before serving requests. Both shell and Zod validate `dataMode` and a root-relative JSON `snapshotUrl`. Cross-origin URLs, traversal, query strings, backslashes and JS injection characters are rejected. Configuration is public and must contain no secrets.

In npm development the Vite middleware serves an empty config script; `VITE_DATA_MODE` and `VITE_LEAKAGE_SNAPSHOT_URL` remain supported. A defined runtime object takes precedence over Vite settings. Invalid runtime settings fail visibly. `copyPublicDir: false` remains mandatory: Vite emits only a harmless default config asset and never copies the runtime directory.

## Verification boundary

On the implementation workstation the Docker CLI is installed but its Linux engine is unavailable. `docker build -t flir-pipeline-explorer .` was attempted and failed to connect to the engine; `docker compose config --quiet` passed. Container startup, HTTP headers, bind-mount behavior and health must be verified with the commands above or the CI Docker job. They are not claimed as locally executed successfully.

The [CI Docker job](https://github.com/Kazzu00/flir-pipeline-explorer/actions/runs/37096999192/job/111128851600) passed: image build, Compose `--wait` with healthy status, SPA routing, runtime configuration, missing-runtime 404, DEMO → REAL without rebuilding, exact synthetic snapshot transport and read-only mount inspection. It then stopped the container. No real artifacts or secrets were used. This remote result does not change the local engine limitation.

References: [Docker multi-stage builds](https://docs.docker.com/build/building/multi-stage/), [nginx try_files](https://nginx.org/en/docs/http/ngx_http_core_module.html#try_files), [Node release policy](https://nodejs.org/en/about/previous-releases).
