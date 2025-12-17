#!/usr/bin/env bash

# 自动重启训练脚本
# 当训练因显存不足(OOM)或其他错误停止时，自动重新启动训练

# 配置参数
CONFIG=${1:-"configs/frnet/frnet-semantickitti_seg.py"}
GPUS=${2:-4}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,2,3,4"}
RESTART_DELAY=${RESTART_DELAY:-30}  # 重启前等待时间（秒），用于清理显存
MAX_RESTARTS=${MAX_RESTARTS:-100}  # 最大重启次数，0表示无限制
LOG_FILE=${LOG_FILE:-"auto_restart_train.log"}  # 日志文件

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 初始化计数器
restart_count=0
start_time=$(date +%s)

# 日志函数
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $1" | tee -a "$LOG_FILE"
}

# 彻底清理函数
cleanup_all_processes() {
    log "开始清理所有相关进程..."
    
    # 查找所有相关进程
    local pids=$(pgrep -f "torch.distributed.launch" 2>/dev/null)
    pids="$pids $(pgrep -f "train.py.*$CONFIG" 2>/dev/null)"
    pids="$pids $(pgrep -f "dist_train.sh.*$CONFIG" 2>/dev/null)"
    
    if [ -n "$pids" ]; then
        log "找到进程: $pids"
        # 先发送SIGTERM，等待优雅退出
        echo "$pids" | xargs -r kill -TERM 2>/dev/null
        sleep 5
        
        # 如果还有进程，强制杀死
        local remaining=$(pgrep -f "torch.distributed.launch\|train.py.*$CONFIG\|dist_train.sh.*$CONFIG" 2>/dev/null)
        if [ -n "$remaining" ]; then
            log "强制清理残留进程: $remaining"
            echo "$remaining" | xargs -r kill -9 2>/dev/null
            sleep 2
        fi
    fi
    
    # 清理端口（如果需要）
    if [ -f "$SCRIPT_DIR/cleanup_port.sh" ]; then
        bash "$SCRIPT_DIR/cleanup_port.sh" 2>/dev/null
    fi
    
    log "进程清理完成"
}

# 清理函数
cleanup() {
    log "收到退出信号，正在清理..."
    cleanup_all_processes
    log "清理完成，退出"
    exit 0
}

# 注册清理函数
trap cleanup SIGINT SIGTERM

log "=========================================="
log "自动重启训练脚本启动"
log "配置文件: $CONFIG"
log "GPU数量: $GPUS"
log "CUDA设备: $CUDA_DEVICES"
log "重启延迟: ${RESTART_DELAY}秒"
log "最大重启次数: ${MAX_RESTARTS:-无限制}"
log "日志文件: $LOG_FILE"
log "=========================================="

# 主循环
while true; do
    # 检查最大重启次数
    if [ "$MAX_RESTARTS" -gt 0 ] && [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
        log "达到最大重启次数 ($MAX_RESTARTS)，停止自动重启"
        break
    fi

    # 如果不是第一次运行，等待一段时间让显存释放
    if [ "$restart_count" -gt 0 ]; then
        log "等待 ${RESTART_DELAY} 秒以释放显存..."
        sleep "$RESTART_DELAY"
        
        # 彻底清理残留进程
        cleanup_all_processes
        sleep 3
    fi
    
    # 每次启动前都检查并清理残留进程（防止之前的进程没有完全退出）
    local existing_pids=$(pgrep -f "torch.distributed.launch\|train.py.*$CONFIG\|dist_train.sh.*$CONFIG" 2>/dev/null)
    if [ -n "$existing_pids" ]; then
        log "检测到残留进程，先清理: $existing_pids"
        cleanup_all_processes
        sleep 3
    fi

    # 记录开始时间
    iteration_start=$(date +%s)
    restart_count=$((restart_count + 1))
    
    log "=========================================="
    log "第 $restart_count 次启动训练"
    log "=========================================="

    # 运行训练命令
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    
    log "执行命令: CUDA_VISIBLE_DEVICES=$CUDA_DEVICES bash dist_train.sh $CONFIG $GPUS --resume"
    
    # 运行训练并捕获退出码
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" bash dist_train.sh "$CONFIG" "$GPUS" --resume
    exit_code=$?
    
    # 计算运行时间
    iteration_end=$(date +%s)
    iteration_duration=$((iteration_end - iteration_start))
    hours=$((iteration_duration / 3600))
    minutes=$(((iteration_duration % 3600) / 60))
    seconds=$((iteration_duration % 60))
    
    log "训练进程退出，退出码: $exit_code"
    log "本次运行时长: ${hours}小时 ${minutes}分钟 ${seconds}秒"
    
    # 检查退出码
    if [ "$exit_code" -eq 0 ]; then
        log "训练正常完成，退出自动重启循环"
        break
    else
        log "训练异常退出（退出码: $exit_code），将在 ${RESTART_DELAY} 秒后重启"
        
        # 立即清理进程，释放资源
        cleanup_all_processes
        
        # 检查是否是OOM错误（可以通过检查日志或退出码判断）
        # 退出码130通常是SIGINT (Ctrl+C)，143是SIGTERM
        if [ "$exit_code" -eq 130 ]; then
            log "可能的原因: 收到SIGINT信号（可能是手动中断）"
        elif [ "$exit_code" -eq 143 ]; then
            log "可能的原因: 收到SIGTERM信号（可能是系统资源限制或手动终止）"
        elif [ "$exit_code" -eq 1 ]; then
            log "可能的原因: 显存不足(OOM)或其他运行时错误"
        else
            log "可能的原因: 未知错误（退出码: $exit_code）"
        fi
    fi
    
    # 显示总运行时间
    total_time=$(($(date +%s) - start_time))
    total_hours=$((total_time / 3600))
    total_minutes=$(((total_time % 3600) / 60))
    log "总运行时间: ${total_hours}小时 ${total_minutes}分钟"
    log ""
done

# 最终统计
total_time=$(($(date +%s) - start_time))
total_hours=$((total_time / 3600))
total_minutes=$(((total_time % 3600) / 60))

log "=========================================="
log "自动重启训练脚本结束"
log "总重启次数: $restart_count"
log "总运行时间: ${total_hours}小时 ${total_minutes}分钟"
log "=========================================="


