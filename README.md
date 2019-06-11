# Vision-based Control of a Quadrotor in User Proximity: Mediated vs End-to-End Learning Approaches
*Dario Mantegazza, Jérôme Guzzi, Luca M. Gambardella and Alessandro Giusti*

Dalle Molle Institute for Artificial Intelligence (IDSIA), USI-SUPSI, Lugano, Switzerland

Proceedings of ICRA 2019

## *Abstract*
We consider the task of controlling a quadrotor
to hover in front of a freely moving user, using input data
from an onboard camera. On this specific task we compare two
widespread learning paradigms: a mediated approach, which
learns an high-level state from the input and then uses it for de-
riving control signals; and an end-to-end approach, which skips
high-level state estimation altogether. We show that despite
their fundamental difference, both approaches yield equivalent
performance on this task. We finally qualitatively analyze the
behavior of a quadrotor implementing such approaches.

<p align="center">
  <img src="/video/gif_github_5.gif"/>
</p>

## Paper Info
[Arxiv](https://arxiv.org/abs/1809.08881) and related [BibTeX](https://github.com/idsia-robotics/proximity-quadrotor-learning/tree/master/bibtex-icra.bib)
## Dataset
The Dataset used is composed of 21 different [rosbag](http://wiki.ros.org/rosbag) files. 

Each rosbag correspond to a single recording session. For the recording sessions we used software developed in house (available [here](https://github.com/jeguzzi/drone_arena)). Each recording session as been manually trimmed to remove system start-up / takeoff / landing phases; this information is available in the `gloabl_parameters.py` script as `bag_start_cut` and `bag_end_cut` dictionaries with the bag name as key.


In each file we recorded multiple [topics](http://wiki.ros.org/Topics); for this paper we use the following topics:

| Topic | Description |
| - | - |
| `/bebop/image_raw/compressed` | Drone's front facing camera feed |
| `/optitrack/head` | Motion Capture information about the 6DOF user's head's [pose](http://docs.ros.org/lunar/api/geometry_msgs/html/msg/Pose.html). In sync with OptiTrack system timestamp |
| `/optitrack/bebop` | Motion Capture information about the 6DOF drone's [pose](http://docs.ros.org/lunar/api/geometry_msgs/html/msg/Pose.html). In sync with OptiTrack system timestamp |
| `/bebop/mocap_odom` | Motion Capture information about the 6DOF drone's [pose](http://docs.ros.org/lunar/api/geometry_msgs/html/msg/Pose.html) + [twist](http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html). In sync with Drone Arena timestamp |
| `/bebop/odom` | Drone's Optical Flow odometry |

In our test we randomly divided the whole dataset in train and test set as follows

| Train set bagfiles | Test set bagfiles |
| - | - |
| 1, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21 | 3, 4, 6, 15 |

The whole dataset can be downloaded (6.6 GB) [here](https://drive.switch.ch/index.php/s/1Q0zN0XDzyRxug4).

A Jupyter notebook implementing dataset extraction from rosbag files can be found [here](https://github.com/idsia-robotics/proximity-quadrotor-learning/tree/master/dataset).

## Code
The code is structured as follows.

    .
    ├── script                  # Script directory
    ├── bagfiles                # Bagfiles main directory
    │   ├── train               # Directory for the bagfiles selected for the train set
    │   └── test                # Directory for the bagfiles selected for the test set
    └── dataset
        ├── version1            # Model 1 dataset .pickle files
        ├── version2            # Model 2 dataset .pickle files
        └── version3            # Model 3 dataset .pickle files

The executable scripts are:
* `dataset_generator.py`
* `keras_train.py`

### `dataset_generator.py`
Create the dataset files used by the models.  After launching the script you will be prompted with a menu in order to select the type of dataset to create.
Each model has its own dataset.

### `keras_train.py`
Uses models (one or all at the same time) for prediction.

<p align="center">
  <img src="/architecture.png"/>
</p>


*Figure: A representation of the three models. Model 1 (left), model 2 (center) and model 3(right).*

All scripts are available [here](https://github.com/idsia-robotics/proximity-quadrotor-learning/tree/master/script).

## Video
The video accepted at ICRA 2019 is available [here](https://github.com/idsia-robotics/proximity-quadrotor-learning/tree/master/video).

## Learning Vision-Based Quadrotor Control in User Proximity
*Dario Mantegazza, Jérôme Guzzi, Luca M. Gambardella and Alessandro Giusti*

The video accepted at HRI 2019 is available [here](https://drive.switch.ch/index.php/s/MlEDrsuHcSl5Aw5) and related BibTeX [here](https://github.com/idsia-robotics/proximity-quadrotor-learning/tree/master/bibtex-hri.bib).

The relative github page is [here](https://github.com/idsia-robotics/proximity-quadrotor-learning-video).

Other videos are available [here](https://drive.switch.ch/index.php/s/hUaJcKEBGbGKDA4).

## Errata
Fixed in v2 (final version for ICRA)<i>: In the paper submission for ICRA2019, each image in Fig.2 have the left and bottom plot with inverted axis. Also in the same figure the smaller plot is rotated by 90° to the right.</i>
