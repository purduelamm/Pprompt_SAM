# Pprompt_SAM
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

Point-based prompting for Segment Anything Model (SAM) that imitates the original demo in [Meta AI](https://segment-anything.com/demo)

<p align="center">
<img src=./images/test/rgb063.png width=30% height=30%> <img src=./images/result/image_pts.png width=30% height=30%> <img src=./images/result/segmented.png width=30% height=30%>
</p>

## Download Process

    git clone https://github.com/kidpaul94/Pprompt_SAM.git
    cd Pprompt_SAM/
    pip3 install -r requirements.txt

## How to Run

> **Note**
`main.py` receives several different arguments. Run the `--help` command to see everything it receives.

     python3 main.py

## Citation
    @inproceedings{kirillov2023segment,
      title={Segment anything},
      author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      pages={4015--4026},
      year={2023}
    }
