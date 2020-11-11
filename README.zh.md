# Detection Delay Index 源代码 (在审)
## 语言
* 默认语言 [en](README.md) 
* 中文简体 [zh](README.zh.md)
## 摘要
在数据流中，**概念漂移(偏离)** 会导致训练数据分布和实际数据分布产生差异，进而使得已经训练好的模型在实际应用中产生较大的偏差。概念偏离检测是一种，主动的，用来检测训练误差和测试误差差异的，信号触发算法，用来提醒机器学习模型，在数据流学习过程中，对差异进行自适应。在目前已有的概念偏离检测算法中，很多是基于模型的预测准确度变化的，即**在某一时间段内，模型的的准确度显著下降，或者，当模型的预测准确度小于给定阈值时，即被认为发生了概念偏离** 。那么如何确定何为显著下降，如何确定阈值，就成为了概念偏离检测的一个核心问题。当前概念偏离检测方法采用了不同的阈值，他们的阈值具有不同意义。那么，如果比较，选择概念偏离检测算法就变得很不清晰。**核心问题在于，不同的概念偏离检测方法，是否可以通过调整概念偏离阈值来达到相同的效果？** 为了解决该问题，我们提出一个全新的指标，Detection Delay Index (DD Index)，用来**量化不同概念偏离检测方法，在不同阈值的条件下，的鲁棒性** 。进而，帮助不同的概念偏离检测方法选择相应的阈值。实验结果显示，通过DD Index选择的阈值，可以使得不同的概念偏离检测算法，实现近乎相同的概念偏离检测精度。并且，我们发现，对于频繁发生概念偏离的数据流，适当降低概念偏离检测鲁棒性会使得整体学习效果更好。

## 作者

* **刘安晋**, Postdoctoral Research Associate, Anjin.Liu@uts.edu.au
* **路节**, Distinguished Professor, Jie.Lu@uts.edu.au
* **宋一辽**, Postdoctoral Research Associate, Anjin.Liu@uts.edu.au
* **宣俊宇**, ARC DECRA, Lecturer, Anjin.Liu@uts.edu.au
* **张广全**, Associate Professor, Guangquan.Zhang@uts.edu.au

Australia Artificial Intelligence Institue, 
Faculty of Engineering and IT
the University of Technology Sydney

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

The work presented in this paper was supported by the Australian Research Council (ARC) under Discovery Project [DP190101733](https://researchdata.edu.au/discovery-projects-grant-id-dp190101733/1378441).

 
