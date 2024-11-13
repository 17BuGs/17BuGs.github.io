---
redirect_from: /_posts/voxel_mamba
title: Voxel Mamba配置过程记录
tags: 经验分享
---


##### 测试环境

- OS：Ubuntu 24.04.1 LTS
- 显卡：NVIDIA GeForce RTX 4090
- CUDA Version：12.4

##### 安装conda环境

python版本是3.8，PyTorch版本2.1.0，对应CUDA版本用的是11.8。(不要用仓库中说的PyTorch 1.12.0，在运行mamba的时候torch会找不到某个模块，应该是torch 2.x才加进去的)

`conda create -n wfmamba python=3.8`

`conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia`

##### clone github仓库

`git clone https://github.com/gwenzhang/Voxel-Mamba.git`

##### 安装spconv

`pip install spconv-cu118`

##### 安装torch_scatter

从网上找到对应CUDA和Python版本的`torch_scatter`的`whl`文件，可能发布了多个版本号，选最新的就可以。我用的是`torch_scatter-2.1.2+pt21cu118-cp38-cp38-linux_x86_64.whl`。下载，然后到下载到的路径下，使用`pip`安装：`pip install torch_scatter-2.1.2+pt21cu118-cp38-cp38-linux_x86_64.whl`

##### 安装OpenPCDet

到`Voxel-Mamba`的主目录下，运行`setup.py`：`python setup.py develop`

出现报错：`error: Couldn't find a setup script in /tmp/easy_install-hh_h_cgj/scikit_image-0.25.0rc1.tar.gz`

先下载`scikit-image`（指定阿里源，我用清华源下得很慢）：`pip install scikit-image -i https://mirrors.aliyun.com/pypi/simple`

又报错：

> ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
> pcdet 0.6.0+ad8172a requires easydict, which is not installed.
> pcdet 0.6.0+ad8172a requires llvmlite, which is not installed.
> pcdet 0.6.0+ad8172a requires numba, which is not installed.
> pcdet 0.6.0+ad8172a requires tensorboardX, which is not installed.

依次安装上面这些库：`pip install easydict llvmlite numba tensorboardX`，之后再安装`scikit-image`，显示已经安装。

之后再运行：`python setup.py develop`，显示：

> Using /home/icdm/anaconda/envs/wfmamba/lib/python3.8/site-packages
> Finished processing dependencies for pcdet==0.6.0+ad8172a`

说明OpenPCDet安装成功了。

##### 安装Mamba

进入到mamba的目录下：`cd mamba`

安装：`pip install -e .`

到此，Voxel Mamba就配置好了。在实际跑代码的时候可能会出现缺一些库的情况，再`pip`装上就好了。
