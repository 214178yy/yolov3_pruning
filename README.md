# yolov3_pruning
## 引言
本仓库参考论文[Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)与[pytorch实现代码](https://github.com/Lam1360/YOLOv3-model-pruning)在tensorflow框架下实现了通道剪枝（gamma正则化）  
根据论文，该剪枝方法将bn层的缩放因子gamma与网络权重进行联合训练，最终通过判断gamma的大小来甄别通道的重要性。  
训练中会对gamma参数进行l1正则化作为惩罚项，使得所有gamma参数都会向0靠拢。
## 使用步骤
### 1.稀疏化训练
运行train_.py文件，可修改参数为gamma惩罚项前的平衡因子，默认为0.01
### 2.剪枝阈值判断
运行judge_thre.py，将会打印出最大剪枝比例，以及实现最大剪枝的阈值。  
通过修改percent参数，可得到期望剪枝比例的所需阈值
### 3.不重要通道置零
运行gamma_zero.py，使得阈值小于参数thre的gamma置0  
根据硬件需求，剪枝后通道数需是16的倍数，修改res_prune处的公式可调整倍数需求  
运行该文件，除了使得原有的ckpt文件中需要置零的gamma为0，还会得到net_channel.json文件，其中包含剪枝后每一层所保留的通道数，该文件将用于剪枝后的模型重建。
### 4.模型重建
运行prune_ckpt.py重建一个剪枝后的压缩模型，实现对gamma为0的通道剪枝。  
输出可进行finetune的ckpt权重文件，以及方便可视化查看剪枝后模型结构的pb文件。
### 5.量化finetune
运行train.py文件进行量化感知训练，实现对模型的finetune以及量化。  
训练前，需要将模型定义的调用改为对稀疏模型定义的调用  
可修改量化参数，包括量化比特数以及多少step后再进行量化的quant_delay  
