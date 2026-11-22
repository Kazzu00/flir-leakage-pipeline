# syntax=docker/dockerfile:1
FROM node:24-alpine@sha256:ebfe2f90462722a7a4de65e91990e97fe0d401c70e0e762c5b53302f905ec1c1 AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY index.html tsconfig.json vite.config.ts ./
COPY src ./src
RUN npm run build

FROM nginx:stable-alpine@sha256:0985e772fb9f729e6fa0980da05fca5d9c468e870eed43071545afa9d2e27d94 AS runtime
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY docker/40-flir-config.sh /docker-entrypoint.d/40-flir-config.sh
RUN sed -i 's/\r$//' /docker-entrypoint.d/40-flir-config.sh && chmod +x /docker-entrypoint.d/40-flir-config.sh
COPY --from=build /app/dist /usr/share/nginx/html
ENV FLIR_DATA_MODE=demo FLIR_LEAKAGE_SNAPSHOT_URL=/runtime/leakage-snapshot.json
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 CMD wget -q -O /dev/null http://127.0.0.1/health || exit 1
