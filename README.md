# monocular-2d-mapping
Final semester project for the *course EL7008: Procesamiento Avanzado de Imágenes*.

The idea is to create simple 2D occupancy grid maps & semantic maps using a monocular camera and semantic segmentation on U-Net.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/gonzal0lguin/monocular-2d-mapping.git
```

2. Clone the official [Panther repository](https://github.com/husarion/panther_ros) and replace the [panther_macro.urdf.xacro](https://drive.google.com/file/d/1Eks7VcxrN7zXWwNEIxzkELbx-4fOxdhI/view?usp=sharing) file from the robot's description.

3. Install ROS dependencies:
```bash
cd ros/dev_ws
sudo rosdep install --from-paths src --ignore-src -y -r
```

4. Compile:
```bash
catkin_make
```

5. Download the [pre-trained model](https://drive.google.com/drive/folders/1kJZWIN_ODuzpFXtbhKbdCHdVQZ3izptV?usp=sharing) and add it to:
```bash
segmentation_models/models/UNet/checkpoints
```

## Usage

To start the simulation and vision node, run:

```bash
source setup.sh
roslaunch gazebo_sim city.launch worldname:=<name-of-world>
```

For Occupancy grid mapping, run:

```bash
rosrun mono_perception mapper
```
You can save the map with:

```bash
rosrun map_server map_saver map:=/global_occ_grid -f <your-map-name>
```

For semantic mapping, run:

```bash
rosrun mono_perception semantic_mapper
```

You can save the map as a gray-scale image with:

```bash
rosrun mono_perception save_sem_map.py
```
