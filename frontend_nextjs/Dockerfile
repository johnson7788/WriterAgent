# 统一主版本，Next 15 建议 Node 20
ARG NODE_VERSION=20

FROM node:${NODE_VERSION}-alpine AS base
WORKDIR /app
ENV CI=1 NEXT_TELEMETRY_DISABLED=1

# 1) 安装依赖
FROM base AS deps
COPY package.json yarn.lock* package-lock.json* pnpm-lock.yaml* ./
RUN set -eux; \
  if [ -f pnpm-lock.yaml ]; then corepack enable && pnpm i --no-frozen-lockfile --registry=http://registry.npm.taobao.org; \
  elif [ -f yarn.lock ]; then yarn --no-frozen-lockfile; \
  elif [ -f package-lock.json ]; then npm ci --registry=http://registry.npm.taobao.org; \
  else echo "Lockfile not found." && exit 1; fi

# 2) 模块解析“早失败”校验
FROM base AS verify
COPY --from=deps /app/node_modules ./node_modules
COPY package.json ./
RUN npm run -s verify:modules

# 3) 质量检查（lint / typecheck）
FROM base AS quality
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run -s lint
RUN npm run -s typecheck

# 4) 生产构建
FROM base AS builder
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run -s build

# 5) 运行镜像（与构建版本一致）
FROM gcr.io/distroless/nodejs20-debian12 AS runner
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
COPY --from=builder /app/next.config.js ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 3000
ENV PORT=3000
CMD ["server.js"]
