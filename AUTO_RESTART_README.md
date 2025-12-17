# 自动重启训练脚本使用说明 (fr_v8)

## 问题描述

训练过程中程序突然收到 **SIGTERM 信号**被终止，可能的原因：
- 系统 OOM Killer（内存不足）
- 残留进程占用资源
- 显存泄漏导致系统资源耗尽

## 快速使用

### 基本用法

```bash
cd /mnt/data0/lqc/fr_v8/my_fr
CUDA_VISIBLE_DEVICES=0,2,3,4 bash auto_restart_train.sh configs/frnet/frnet-semantickitti_seg.py 4
```

### 在后台运行

```bash
# 使用nohup在后台运行
nohup CUDA_VISIBLE_DEVICES=0,2,3,4 bash auto_restart_train.sh configs/frnet/frnet-semantickitti_seg.py 4 > training.log 2>&1 &

# 或使用screen/tmux
screen -S training
CUDA_VISIBLE_DEVICES=0,2,3,4 bash auto_restart_train.sh configs/frnet/frnet-semantickitti_seg.py 4
# 按 Ctrl+A 然后 D 来detach
```

## 脚本功能

1. **自动检测退出**: 检测训练进程的退出码，非0退出码会触发重启
2. **显存清理**: 每次重启前等待指定时间（默认30秒）以释放显存
3. **进程清理**: 自动清理残留的训练进程（包括子进程）
4. **日志记录**: 所有操作都会记录到 `auto_restart_train.log`
5. **正常退出**: 如果训练正常完成（退出码0），脚本会自动停止
6. **智能识别**: 区分不同类型的退出（SIGTERM、SIGINT、OOM等）

## 环境变量配置

```bash
# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,2,3,4

# 设置重启前等待时间（秒，默认30秒）
export RESTART_DELAY=60

# 设置最大重启次数（默认100，0表示无限制）
export MAX_RESTARTS=50

# 设置日志文件路径（默认auto_restart_train.log）
export LOG_FILE=my_training.log

# 运行脚本
bash auto_restart_train.sh configs/frnet/frnet-semantickitti_seg.py 4
```

## 清理残留进程

如果发现大量残留进程，使用清理脚本：

```bash
# 清理所有训练相关进程
bash cleanup_training_processes.sh

# 或指定配置文件
bash cleanup_training_processes.sh configs/frnet/frnet-semantickitti_seg.py
```

## 退出码说明

| 退出码 | 含义 | 可能原因 |
|--------|------|----------|
| 0 | 正常退出 | 训练完成 |
| 1 | 一般错误 | OOM、运行时错误 |
| 130 | SIGINT | Ctrl+C中断 |
| 143 | SIGTERM | 系统终止、资源限制（你遇到的情况） |
| 137 | SIGKILL | 强制终止（OOM Killer） |

## 常见问题

### Q1: 程序频繁重启怎么办？

**A**: 可能是显存不足，建议：
1. 增加重启延迟：`export RESTART_DELAY=60`
2. 减少显存占用（参考配置优化文档）
3. 检查系统内存：`free -h`

### Q2: 如何停止脚本？

**A**: 
```bash
# 找到脚本进程ID
ps aux | grep auto_restart_train.sh

# 发送SIGTERM信号
kill <PID>
```

### Q3: 如何查看日志？

**A**: 
```bash
# 查看自动重启日志
tail -f auto_restart_train.log

# 查看训练日志（在work_dirs目录下）
tail -f work_dirs/frnet-semantickitti_seg/*/latest.log
```

### Q4: 脚本一直重启，训练无法进行？

**A**: 可能是配置问题，建议：
1. 先手动运行一次训练，检查是否有配置错误
2. 清理残留进程：`bash cleanup_training_processes.sh`
3. 检查GPU显存：`nvidia-smi`

## 诊断步骤

如果问题持续存在：

1. **检查系统内存**:
   ```bash
   free -h
   ps aux --sort=-%mem | head -20
   ```

2. **检查GPU显存**:
   ```bash
   nvidia-smi
   watch -n 1 nvidia-smi
   ```

3. **检查残留进程**:
   ```bash
   ps aux | grep -E "train.py|dist_train|torch.distributed" | grep -v grep
   ```

4. **查看系统日志**（需要root权限）:
   ```bash
   sudo dmesg | grep -i "killed\|oom\|memory" | tail -20
   ```

## 注意事项

1. **显存释放**: 如果显存释放较慢，可以增加 `RESTART_DELAY` 的值
2. **端口占用**: 如果遇到端口占用问题，脚本会自动使用 `cleanup_port.sh` 清理
3. **日志文件**: 长时间运行会产生较大的日志文件，建议定期清理
4. **正常训练**: 如果训练正常完成，脚本会自动退出，不会无限重启

## 示例输出

```
[2025-12-13 10:30:00] ==========================================
[2025-12-13 10:30:00] 自动重启训练脚本启动
[2025-12-13 10:30:00] 配置文件: configs/frnet/frnet-semantickitti_seg.py
[2025-12-13 10:30:00] GPU数量: 4
[2025-12-13 10:30:00] CUDA设备: 0,2,3,4
[2025-12-13 10:30:00] 重启延迟: 30秒
[2025-12-13 10:30:00] ==========================================
[2025-12-13 10:30:00] 第 1 次启动训练
[2025-12-13 10:30:00] 执行命令: CUDA_VISIBLE_DEVICES=0,2,3,4 bash dist_train.sh configs/frnet/frnet-semantickitti_seg.py 4 --resume
...
[2025-12-13 10:52:37] 训练进程退出，退出码: 143
[2025-12-13 10:52:37] 本次运行时长: 0小时 22分钟 37秒
[2025-12-13 10:52:37] 训练异常退出（退出码: 143），将在 30 秒后重启
[2025-12-13 10:52:37] 可能的原因: 收到SIGTERM信号（可能是系统资源限制或手动终止）
[2025-12-13 10:53:07] 等待 30 秒以释放显存...
[2025-12-13 10:53:07] 开始清理所有相关进程...
[2025-12-13 10:53:12] 进程清理完成
[2025-12-13 10:53:15] 第 2 次启动训练
...
```


