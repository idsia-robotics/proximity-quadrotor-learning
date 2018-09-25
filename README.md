# Vision-based Control of a Quadrotor in User Proximity: Mediated vs End-to-End Learning Approaches
*Dario Mantegazza, Jérôme Guzzi, Luca M. Gambardella and Alessandro Giusti*
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
### Dataset
The Dataset used is composed of 21 different [rosbag](http://wiki.ros.org/rosbag) files. 

Each rosbag correspond to a single recording session. For the recording sessions we used the [Drone Arena](https://github.com/jeguzzi/drone_arena) software.

In each file we recorded multiple [topics](http://wiki.ros.org/Topics); for this paper we use the following topics:

| Topic | data contained |
| - | - |
| `/bebop/image_raw/compressed` | Drone's front facing camera feed |
| `/optitrack/head` | Motion Capture information about the 6DOF user's head's [pose](http://docs.ros.org/lunar/api/geometry_msgs/html/msg/Pose.html). In sync with OptiTrack system timestamp |
| `/optitrack/bebop` | Motion Capture information about the 6DOF drone's [pose](http://docs.ros.org/lunar/api/geometry_msgs/html/msg/Pose.html). In sync with OptiTrack system timestamp |
| `/bebop/mocap_odom` | Motion Capture information about the 6DOF drone's [pose](http://docs.ros.org/lunar/api/geometry_msgs/html/msg/Pose.html) + [twist](http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html). In sync with Drone Arena timestamp |
| `/bebop/odom` | Drone's Optical Flow odometry |


The whole dataset can be downloaded [here](https://drive.switch.ch/index.php/s/1Q0zN0XDzyRxug4).

A Jupyter notebook on how we extract the data can be found TODO.

### Code
    .
    ├── script
    ├── bagfiles                # Bagfiles main dir
    │   ├── train               # Directory for the bagfiles selected for the train set
    │   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
    │   └── unit                # Unit tests
    └── ...
TODO [here](https://github.com/idsia-robotics/proximity-quadrotor-learning/tree/master/script)

### Video
TODO
[here](https://github.com/idsia-robotics/proximity-quadrotor-learning/tree/master/video)

### Errata
TODO
