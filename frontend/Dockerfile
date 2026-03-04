# QuantFlow — Next.js frontend image
#
# Build context: ./frontend
#
# Multi-stage build:
#   deps    — install node_modules (cached layer)
#   builder — run next build
#   runner  — minimal production image (standalone output)

# ---------------------------------------------------------------------------
# Stage 1: install dependencies
# ---------------------------------------------------------------------------
FROM node:20-alpine AS deps

WORKDIR /app

COPY package.json package-lock.json ./

# Install ALL deps (devDependencies like tailwindcss are needed for build)
RUN npm ci

# ---------------------------------------------------------------------------
# Stage 2: build
# ---------------------------------------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production

RUN npm run build

# ---------------------------------------------------------------------------
# Stage 3: production runner
# ---------------------------------------------------------------------------
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV HOSTNAME="0.0.0.0"

RUN addgroup -S quantflow && adduser -S quantflow -G quantflow

# Copy standalone server + static assets
COPY --from=builder /app/public ./public
COPY --from=builder --chown=quantflow:quantflow /app/.next/standalone ./
COPY --from=builder --chown=quantflow:quantflow /app/.next/static ./.next/static

USER quantflow

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD wget -q -O - http://localhost:3000/ || exit 1

CMD ["node", "server.js"]
