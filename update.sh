#!/usr/bin/env bash
#
# 部署更新脚本 —— 拉取最新代码、重建镜像、重启容器（等待健康检查通过）
#
# 用法：./update.sh
#
# 网段在 .env 里配置：WB_NETWORK=mynet
# 前提：上游同步与冲突解决在开发机完成并推送到 fork，
#       本机只需拉取 fork 的最新 main 再重新部署。
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

COMPOSE_FILE="docker-compose.local.yml"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }

# 兼容 docker compose (v2) 与 docker-compose (v1)
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  echo "[x] 未找到 docker compose" >&2
  exit 1
fi

log "拉取最新代码"
git pull --ff-only

# 支持 --wait 的 compose 会等到容器健康再返回；老版本则降级为不等待
# （grep 不用 -q：-q 会提前退出触发 SIGPIPE，pipefail 下会误判）
WAIT_FLAGS=""
if $COMPOSE up --help 2>&1 | grep -- '--wait' >/dev/null; then
  WAIT_FLAGS="--wait --wait-timeout 180"
else
  warn "当前 compose 不支持 --wait，将不等待健康状态"
fi

log "重建镜像并重启容器"
# shellcheck disable=SC2086
if $COMPOSE -f "$COMPOSE_FILE" up -d --build $WAIT_FLAGS; then
  ok "容器已就绪"
else
  warn "容器未在超时内变为健康，查看日志："
  echo "    $COMPOSE -f $COMPOSE_FILE logs --tail 50"
  exit 1
fi

echo
log "当前状态"
$COMPOSE -f "$COMPOSE_FILE" ps
