# Next.js dashboard, built as a standalone server.
#   docker build -t quantlab-web --build-arg NEXT_PUBLIC_API_URL=http://localhost:8000 web

FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# NEXT_PUBLIC_* is inlined into the browser bundle at build time, so it has to be a build arg.
# It is the URL the *browser* uses, i.e. the API port published on the host.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:22-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN adduser -D -u 10001 quantlab
# standalone output = server.js + only the node_modules it needs
COPY --from=build --chown=quantlab /app/.next/standalone ./
COPY --from=build --chown=quantlab /app/.next/static ./.next/static
COPY --from=build --chown=quantlab /app/public ./public
USER quantlab

EXPOSE 3000
CMD ["node", "server.js"]
