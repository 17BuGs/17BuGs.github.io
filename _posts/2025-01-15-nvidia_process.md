---
redirect_from: /_posts/nvidia_process
title: 记一次清理CUDA上的僵尸进程
tags: 经验分享
---


##### 背景

某一次炼丹时，使用`nvidia-smi`，发现显存占用和显示的进程对不上，有大约`7000MiB`的占用不知道在哪里。

##### 解决过程

使用`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`，可以看到详细的`pid`进程号和占用的显存。输出：

> pid, used_gpu_memory [MiB]
> 4135549, 7242 MiB

但是当使用`kill -9 4135549`，输出：

> bash: kill: (4135549) - 没有那个进程

使用`ps -aux | grep 4135549`，依然不能找到对应的进程。

最后，使用命令`pkill -u [username] python`，杀掉指定用户的所有`python`程序。使用`nvidia-smi`，发现被占用的显存回来了。

完。

##### 复盘

可能是VSCode之前崩过一次，留下的僵尸进程。
