<img src="./assets/HEAD-icon.jpg" alt="HEAD icon" style="display:block; margin: 0 auto; width: 400px;">

# HEAD:Holistic Evolutionary Autonomous Driving
HEAD is a holistic suite of evolutionary autonomous driving software, based on the MetaDrive simulation platform, that seamlessly imports driving scenarios, uploads training models, and efficiently performs continuous training designed to significantly improve the performance of arbitrary models.
## Introduction
**HEAD (Holistic Evolutionary Autonomous Driving)** is an Autonomous Driving Platform with the following key features: 
- **A General Self-Evolutionary Autonomous Driving Software Tool**: It combines learning-based, optimization-based, and rule-based algorithms to efficiently handle complex driving scenarios and ensure safety and performance.
- **Integration with Simulation Testing**: It is deeply integrated with the MetaDrive simulation platform, enabling comprehensive testing and optimization.
- **A Closed-Loop Data-Driven Platform**: It provides a complete closed-loop system from scenario generation to algorithm evolution, enhancing adaptability and reliability in unseen scenarios through adversarial testing and continuous learning.
![](./assets/HEAD.jpg)
## 🔧Quick Start
1. **Clone the repo**

   Start by cloning the HEAD repository to your local machine:
    ``` bash
    git clone https://github.com/TJHuangteam/HEAD.git
    cd HEAD
   ```
2. **Conda Env Settings and Install Dependencies**
    ``` bash
    conda create -n HEAD python=3.9
    conda activate HEAD
    pip install -r requirements.txt
    ```
   



## References

If you use HEAD in your own work, please cite:
```text
@article{yang2024guarantee,
  title={How to guarantee driving safety for autonomous vehicles in a real-world environment: a perspective on self-evolution mechanisms},
  author={Yang, Shuo and Huang, Yanjun and Li, Li and Feng, Shuo and Na, Xiaoxiang and Chen, Hong and Khajepour, Amir},
  journal={IEEE Intelligent Transportation Systems Magazine},
  year={2024},
  publisher={IEEE}
}
```





## Acknowledgements

Github:[GitHub - metadriverse/metadrive: MetaDrive: Open-source driving simulator](https://github.com/metadriverse/metadrive)

Website:[MetaDrive | MetaDriverse](https://metadriverse.github.io//metadrive/)



``` text
@article{li2021metadrive,
  title={MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning},
  author={Li, Quanyi and Peng, Zhenghao and Xue, Zhenghai and Zhang, Qihang and Zhou, Bolei},
  journal={arXiv preprint arXiv:2109.12674},
  year={2021}
}
```



## Relevant Projects

**Metadrive: Composing diverse driving scenarios for generalizable reinforcement learning**
\
Li, Quanyi and Peng, Zhenghao and Feng, Lan and Zhang, Qihang and Xue, Zhenghai and Zhou, Bolei
\
*IEEE Transactions on Pattern Analysis and Machine Intelligence*
\
[
<a href="https://arxiv.org/pdf/2109.12674.pdf">Paper</a>
|
<a href="https://metadriverse.github.io/metadrive-simulator/">Website</a>
|
<a href="https://github.com/metadriverse/metadrive">Code</a>
]




## License

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.



