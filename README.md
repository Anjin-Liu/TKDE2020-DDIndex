# Source Code for Detection Delay Index (under review)
## language
* Default [en](README.md) 
* Optional [zh](README.zh.md)
## Abstract
In data stream learning, changes in the data’s distribution can undermine model performance. Concept drift detection is a proactive way of tracking these changes to signal that the model needs to update. Many methods use thresholds based on models’ error rates to detect concept drift. However, selecting drift thresholds and detection mechanics that never truly results in a clear picture of what contribute to the stream learning process the most. To bring some clarity to this process, we developed recursive drift threshold searching algorithm to align the drift thresholds of a set of algorithms so they all at the same robustness level. As a result, threshold selection is removed as a factor, leaving clear space to focus on the drift detection mechanics themselves. From a series of experiments, we provide a novel set of insights into how drift detection algorithms perform in different drift scenarios. We find that, for data streams with frequent drifts, a higher detection sensitivity improves accuracy. Moreover, the experiment results reveal that drift thresholds should not be fixed during stream learning. Rather, they should adjust dynamically based on the prevailing conditions of the data stream.

## Authors

* **Anjin Liu** Postdoctoral Research Associate, Anjin.Liu@uts.edu.au
* **Jie Lu** Distinguished Professor, Jie.Lu@uts.edu.au
* **Yiliao Song** Postdoctoral Research Associate, Anjin.Liu@uts.edu.au
* **Junyu Xuan** ARC DECRA, Lecturer, Anjin.Liu@uts.edu.au
* **Guangquan Zhang** Associate Professor, Guangquan.Zhang@uts.edu.au

Australia Artificial Intelligence Institue, 
Faculty of Engineering and IT
the University of Technology Sydney

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

The work presented in this paper was supported by the Australian Research Council (ARC) under Discovery Project [DP190101733](https://researchdata.edu.au/discovery-projects-grant-id-dp190101733/1378441).

 
