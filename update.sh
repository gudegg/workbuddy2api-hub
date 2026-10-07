#!/usr/bin/env bash
#
# 部署更新脚本 —— 拉取最新代码、重建镜像、重启容器（等待健康检查通过）
#
# 用法：./update.sh
#
# 网段在 .env 里配置（WB_NETWORK=xxx）；.env 不入库（已 gitignore），
# 首次运行会自动从 .env.example 生成。
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

COMPOSE_FILE="docker-compose.local.yml"
SERVICE="wb-proxy"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

# 兼容 docker compose (v2) 与 docker-compose (v1)
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  die "未找到 docker compose（v1 或 v2）"
fi

log "拉取最新代码"
git pull --ff-only

# 首次运行：从 .env.example 生成 .env（.env 不入库，可自由改）
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  warn "已从 .env.example 生成 .env；如需改网段请编辑 .env"
fi

log "构建镜像"
$COMPOSE -f "$COMPOSE_FILE" build || die "镜像构建失败，请检查源码与网络"

# 支持 --wait 的 compose 会等到容器健康再返回
WAIT_FLAGS=""
if $COMPOSE up --help 2>&1 | grep -- '--wait' >/dev/null; then
  WAIT_FLAGS="--wait --wait-timeout 180"
fi

log "启动容器"
# shellcheck disable=SC2086
if ! $COMPOSE -f "$COMPOSE_FILE" up -d $WAIT_FLAGS; then
  warn "容器启动失败，或未在超时内变为健康。查看日志："
  echo "    $COMPOSE -f $COMPOSE_FILE logs --tail 50"
  exit 1
fi

# 老版 compose 无 --wait：手动轮询健康，避免「假成功」
if [ -z "$WAIT_FLAGS" ]; then
  warn "当前 compose 不支持 --wait，改为手动轮询健康状态…"
  cid="$($COMPOSE -f "$COMPOSE_FILE" ps -q "$SERVICE")"
  [ -n "$cid" ] || die "未找到容器，启动可能失败"
  ready=0
  for _ in $(seq 1 36); do
    st="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}pending{{end}}' "$cid" 2>/dev/null || echo missing)"
    if [ "$st" = "healthy" ]; then ready=1; break; fi
    case "$st" in
      unhealthy|missing)
        die "容器状态异常（$st），查看日志：$COMPOSE -f $COMPOSE_FILE logs --tail 50" ;;
    esac
    sleep 5
  done
  [ "$ready" = 1 ] || die "健康检查超时（180s），查看日志：$COMPOSE -f $COMPOSE_FILE logs --tail 50"
fi

ok "容器已就绪"
echo
$COMPOSE -f "$COMPOSE_FILE" ps
