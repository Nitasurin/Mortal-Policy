# Mortal-Policy

This repository is a branch of  [**Mortal original repository**](https://github.com/Equim-chan/Mortal) ,transitioning from value-based methods to policy-based methods.   


## Overview
Initially developed in 2022 based on Mortal V2, migration of modified code to Mortal V4 in 2024   
this branch features:
- More stable performance optimization process
- Enhanced asymptotic performance

## Installation
Consistent with the original repository. Read the [**Documentation**](https://mortal.ekyu.moe)  
**Torch requirement:** torch2.5.1+cu124 (install via pip)

## Run
Mortal-Policy adopts an **offline to online** training approach:

1. **Data Preparation**  
   Collect samples in `mjai` format.

2. **Configuration**  
   Rename `config.example.toml` to `config.toml` and set hyperparameters.

3. **Training Stages**  
   - *Offline Phase (Advantage Weighted Regression):*  
     Run `train_offline.py`
   - *Online Phase (Policy Gradient with Importance Sampling and PPO-style Clipping):*  
     Run `train_online.py`

⚠️ **Important Note**  
**Advantage Weighted Regression(AWR)** is not included in the original implementation based on Mortal V2. You can try the following alternative options: **Behavior Cloning(BC)**, **Behavior Proximal Policy Optimization (BPPO)**, or **distillation** from the value-based Mortal.    
While online-only training is possible, it is **not recommended**.This branch omits critical online training features: Opponent Pool, Policy Archive, etc.

## Weights & Configuration
Maintained alignment with original Mortal repository. For details see [this post](https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56).   
**The weights, hyperparameters, and some online training features have been removed from this branch when it was open-sourced.** 


## License
### Code
[![AGPL-3.0-or-later](https://github.com/Equim-chan/Mortal/raw/main/docs/src/assets/agpl.png)](https://github.com/Equim-chan/Mortal/blob/main/LICENSE)

Copyright (C) 2021-2022 Equim  
Copyright (C) 2025 Nitasurin

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

### Assets
[![CC BY-SA 4.0](https://github.com/Equim-chan/Mortal/raw/main/docs/src/assets/by-sa.png)](https://creativecommons.org/licenses/by-sa/4.0/)
