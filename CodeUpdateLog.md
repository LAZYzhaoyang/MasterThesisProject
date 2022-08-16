# Code Update Log
author: Zhaoyang Li
Central South University

### 2022/08/15
version 3.1 修补了3.0版本的一些漏洞，所有代码均上传至GitHub，在3.1版本之后所有的代码将会与数据，结果分隔，不再放在同一文件夹，代码更规范，更易修改和使用。

本次更新主要修改了utils.trainlib中的代码，对所有的训练函数进行了一个整合，统一通过main_train函数调用，并支持修改很多超参数，如需进一步添加可调节的超参数可以通过修改get_task_config函数实现。

### 2022/08/15
version 3.0 第一个完整的全流程版本，包括了模型训练和管子全流程优化程序。目前优化部分程序需要进一步慢慢调优，包括后续结果的输出和保存等。接下来的工作主要转入论文撰写，程序部分的工作基本完成，后续工作主要为论文撰写提供图片和结果。

### 2022/08/10
version 2.8 更新了pretrain-KMeans方法，完成了从载入，保存，训练，预测等全流程的对象化实现，支持多cluster的聚类实现。

### 2022/08/08
version 2.7 更新了spice方法，目前没有测试其性能，但已完成debug，可以使用，应该会有一定的提升

接下来将进行全流程优化实现，且开始写大论文，不管聚类效果如何均就这样了

### 2022/08/01
version 2.6 更新了自制的DeepCluster方法，目前方法对于预训练模型没有提升效果，接下来将直接进入下一阶段，将优化全流程实现，先暂时不管这部分的效果。

### 2022/07/30
version 2.5 更新了byol，simsiam的预训练方法，并构建了pretrain-kmeans的聚类效果，支持导入预训练backbone或者supervised的backbone

目前预训练效果仍然很差，需要想办法进一步改进，下一步考虑添加DeepCluster方法，并开始整合所有代码进行吸能结构优化。
主要任务，构建整个全流程的代码框架。

### 2022/07/23
version 2.4更新了scan深度聚类算法程序，效果很差，可能需要从预训练和scan算法本身进一步改进提升

下一步尝试使用BYOL进行预训练

### 2022/07/16
version 2.3 更新了simclr预训练程序，不过目前效果一般，需要进一步提升

### 2022/07/07

version 2.2 版本算是2.0版本以来的第一个大改版，解决了一个痛点问题。即在之前的版本中都未解决的训练集准确率已达到100但测试集准确率只有70不到的问题。以下将对问题和解决思路进行详细讲解：

出现的问题：
训练时发现无论如何划分训练集和测试集，模型在测试集上的准确率难以超过75，尝试了调整学习率，优化器以及batch size等常规做法，均无法奏效。

解决历程：
尝试将展示模型在训练集上的效果，发现模型在训练集上很快收敛至100准确率并不再更新参数

推测有几个原因，数据标签出错导致训练集测试集的分布存在较大偏差，使得模型泛化能力下降；或是数据量过少，使得模型过快在训练集上过拟合，进而降低模型泛化能力；模型缺少正则项。

首先添加了dropout层用于实现正则化，发现模型在训练集上收敛速度确实有所下降，但仍然很快就到达100准确率难以进一步优化。

其次检查数据标签，发现确实存在较多标错情况，于是花时间重新进行了统一的标注，使用标注后的数据训练，在测试集上的准确率有所提升，但并不大，大约是75到80左右。

最后对原始数据进行更复杂的数据增强操作，增加数据的多样性，详情见./data_loader/data_aug.py文件。具体是用get_transformer函数中输出的三种transform中的weak_augment替换了在SupervisedDataset中使用的standard。

结合三种方法，使得模型在训练集和测试集上的准确率接近，并且能够随着训练进行稳定提升至94%左右。基本解决supervised任务，只是目前训练速度有待提高，接下来的工作重心将转移至simclr任务和scan, spice任务中，期间也会进行一些提速方面的测试。


### 2022/07/05

Thesis Code 2.0

对1.x版本的代码进行了全新重构，将重复功能的代码整合，整个代码框架相较第一代来说更简洁易懂，并且核心功能和性能并没有下降。反而有所提升。工具类函数统一放在utils下，分为datalib, evaluationlib, losslib, modellib, plotlib, toollib, trainlib几部分，分别用于处理数据，评估模型，构建损失函数，构建模型，画图，工具类，训练类函数等。

目前的代码仍然未完善，只实现了Response部分的训练函数，supervised任务的代码几近完成，但存在测试集准确率难以提升的问题，目前正在查找问题中。moco, simclr等pretext任务和scan, spice等深度聚类任务的函数仍未完成重构，将在接下来的版本中进行更新。