原项目的组织有点混乱，想运行要改很多路径，而且分割不是很明确，这份代码重新组织整理了一下结构

### 目录职能:

```shell
├─DataPreparation	# 预处理数据代码
├─Dataset			# 数据存访处，包括原始数据和预处理数据
├─ModelDeduction	# 模型推演
├─Models			# 存访模型，可以保存用于复现效果的模型
├─ModelTrain		# 模型训练的代码
└─Runs				# 存放tensorboard日志记录
```

