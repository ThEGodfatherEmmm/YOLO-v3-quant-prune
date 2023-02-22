## 文件结构及环境说明
[main.py](main.py) 用于训练的文件 --cfg参数用于有cfg的模型加载及保存时
[trtmodel.py](trtmodel.py) 用于将模型转换为TensorRT模型的文件
[vggprune.py](vggprune.py) 用于执行剪枝的文件
[vggquant.py](vggquant.py) 用于执行量化的文件
[quant.py](quant.py) 用于保存量化方法的文件
data.cifar10 保存cifar10数据集的文件夹
logs 保存默认vgg训练模型的文件夹
logs_prune 保存剪枝后模型的文件夹
logs_prune_refine 保存剪枝及重训练后模型的文件夹
logs_quant 保存量化后模型的文件夹
trt 文件夹中保存了jetson nano开发板上下载的各类trt模型
[quant_0.9refine_log.txt](quant_0.9refine_log.txt) 保存了90%的剪枝之后重训练之后不同方法下量化之后的输出
[quant_log.txt](quant_log.txt) 保存了不同方法下量化之后的输出

本次实验除了TensorRT的部署相关内容都在windows上的PyTorch环境下完成，TensorRT的部署相关内容在jetson nano上进行

## 运行方法
main.py和vggquant.py得到的模型默认不含cfg信息，vggprune.py生成的模型含有cfg信息，重训练时需要添加--cfg参数，得到的模型含有cfg信息，可以被torch2trt转换为TensorRT模型
