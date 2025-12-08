#!/usr/bin/env bash
# 清理占用指定端口的进程

PORT=${1:-29500}

echo "正在查找占用端口 $PORT 的进程..."

# 查找占用端口的进程PID
PIDS=$(lsof -ti :$PORT 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "端口 $PORT 未被占用"
    exit 0
fi

echo "找到以下进程占用端口 $PORT:"
ps -p $PIDS -o pid,user,cmd

read -p "是否要终止这些进程? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "正在终止进程..."
    kill -9 $PIDS
    sleep 1
    echo "进程已终止"
else
    echo "操作已取消"
fi

