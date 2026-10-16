#!/bin/bash
set -e

# Source ROS 2 and the local workspace
source /opt/ros/humble/setup.bash
source /home/ubuntu/mcl_ws/install/setup.bash

exec "$@"
