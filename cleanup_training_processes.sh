#!/usr/bin/env bash

# 清理所有训练相关进程的脚本

CONFIG=${1:-"configs/frnet/frnet-semantickitti_seg.py"}

echo "正在查找并清理所有训练相关进程..."

# 查找所有相关进程
pids=$(pgrep -f "torch.distributed.launch" 2>/dev/null)
pids="$pids $(pgrep -f "train.py.*$CONFIG" 2>/dev/null)"
pids="$pids $(pgrep -f "dist_train.sh.*$CONFIG" 2>/dev/null)"
pids="$pids $(pgrep -f "auto_restart_train.sh.*$CONFIG" 2>/dev/null)"

if [ -z "$pids" ]; then
    echo "没有找到相关进程"
    exit 0
fi

echo "找到以下进程:"
echo "$pids" | tr ' ' '\n' | while read pid; do
    if [ -n "$pid" ]; then
        ps -p "$pid" -o pid,cmd --no-headers 2>/dev/null
    fi
done

echo ""
read -p "是否要终止这些进程? (y/N): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "取消操作"
    exit 0
fi

# 先发送SIGTERM
echo "发送SIGTERM信号..."
echo "$pids" | tr ' ' '\n' | while read pid; do
    if [ -n "$pid" ]; then
        kill -TERM "$pid" 2>/dev/null && echo "已发送SIGTERM到进程 $pid"
    fi
done

sleep 5

# 检查是否还有残留进程
remaining=$(pgrep -f "torch.distributed.launch\|train.py.*$CONFIG\|dist_train.sh.*$CONFIG\|auto_restart_train.sh.*$CONFIG" 2>/dev/null)

if [ -n "$remaining" ]; then
    echo "检测到残留进程，强制终止..."
    echo "$remaining" | tr ' ' '\n' | while read pid; do
        if [ -n "$pid" ]; then
            kill -9 "$pid" 2>/dev/null && echo "已强制终止进程 $pid"
        fi
    done
    sleep 2
fi

# 清理端口
if [ -f "$(dirname "$0")/cleanup_port.sh" ]; then
    echo "清理端口..."
    bash "$(dirname "$0")/cleanup_port.sh"
fi

echo "清理完成"


