# GBDT_LR
## 概述
* 采用sklearn实现GBDT，获取其叶子结点输出，生成离散特征，再进行lr模型的训练
* 暂不支持batch训练，配置文件中batch_size需设置为：0 。因为sklearn中的GBDT 能够设置树的个数，每棵树最大叶子节点个数等超参数，但不能指定每颗树的叶子节点数，导致每个batch生成的离散特征维度不一致
，numpy数组无法拼接；可以借助lightgbm 实现，需要安装相关包

## 运行代码
### Setup
```
cd conf
vim feature_conf.ini
vim model_conf.ini
```

### Train
设置model_conf中的mode：train<br>
　　　　　　　　　data_dir_train: ...
```
cd conf
vim model_conf.ini
```
```
cd python
python train.py
```

### Pred
设置model_conf中的mode：pred<br>
　　　　　　　　　data_dir_pred: ...
```
cd conf
vim model_conf.ini
```
```
cd python
python train.py
```
