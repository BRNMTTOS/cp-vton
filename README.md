# 复现 & 思考

	源文件 Readme-raw.md

## 1. 环境搭建 & 配置错误

**环境**

```
$ pip list | grep torch
torch               1.1.0
torchvision         0.3.0
```

**tensorboardX**

```
Traceback (most recent call last):
  File "train.py", line 12, in <module>
    from tensorboardX import SummaryWriter
ModuleNotFoundError: No module named 'tensorboardX'

$ pip install tensorboardX

```


**修改代码 cp_dataset.py, tranform设置错误**

```
Traceback (most recent call last):
  File "train.py", line 191, in <module>
    main()
  File "train.py", line 176, in main
    train_gmm(opt, train_loader, model, board)
  File "train.py", line 58, in train_gmm
    inputs = train_loader.next_batch()
  File "/app/project/vton/cinastanbean-cp-vton/cp_dataset.py", line 166, in next_batch
...
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
RuntimeError: output with shape [1, 256, 192] doesn't match the broadcast shape [3, 256, 192]

```

**--workers 4 --> --workers 0**

```
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
...
RuntimeError: DataLoader worker (pid 2841) is killed by signal: Bus error.
```


## 2. Train & Test



## 3. Virtual Try-On 技术路线的瓶颈

虚拟模特图像生成，技术上大致有三条路实现。

“Virtual Try-On”（VTON）是其中一种方式。

VTON技术有如下考虑：
1. 规避模特生成问题，模特生成本身比较难以做到，难以做到对模特面孔头发、身材真实性等方面的保真度，VTON技术路线规避该问题；
2. 默认模特已经穿着了和待合成服饰尺寸形状大体一致的服饰，通过对服饰做Warping进而“贴图”，实现Try-On的效果。


技术产品化VTON思路还有些问题：
 1. 对指定模特，给他换上另外一套衣服，版权问题；
 2. 服装和人的搭配问题，如何保持视觉协调；
 3. 服装穿着在人身上产生的自然形变，因为对服饰做Warping没有根本解决对服饰的理解问题。

 
 
 