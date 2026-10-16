###############################################################################
# MCL Particle Filter – ROS 2 Humble Docker image
###############################################################################
FROM osrf/ros:humble-desktop-full AS base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-scipy \
    python3-matplotlib \
    python3-opencv \
    ros-humble-nav2-map-server \
    ros-humble-nav2-lifecycle-manager \
    ros-humble-nav2-amcl \
    ros-humble-tf2-ros \
    ros-humble-rviz2 \
    x11-apps \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy scipy

# ── Create workspace ────────────────────────────────────────────────────────
RUN mkdir -p /home/ubuntu/mcl_ws/src
WORKDIR /home/ubuntu/mcl_ws

# Copy source code
COPY src/ src/

# ── Build ROS 2 workspace ───────────────────────────────────────────────────
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --packages-select mcl_localization mcl_robot

# ── Copy additional project files ───────────────────────────────────────────
COPY maps/     maps/
COPY plots/    plots/
COPY bridge/   bridge/
COPY launch/   launch/
COPY rviz/     rviz/
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/fastdds_no_shm.xml /fastdds_no_shm.xml
RUN chmod +x /entrypoint.sh

# Force Fast DDS to use UDP only (no shared memory) —
# required for inter-container communication with host networking
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds_no_shm.xml

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
