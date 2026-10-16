#!/usr/bin/env bash
###############################################################################
# run.sh — Build and launch the MCL Particle Filter demo
#
# Usage:
#   ./run.sh              # build + launch everything (RViz, robot, MCL, plots)
#   ./run.sh build        # only build the Docker image
#   ./run.sh up           # only start containers (skip build)
#   ./run.sh down         # stop all containers
#   ./run.sh shell        # open a bash shell inside the container
#   ./run.sh launch       # run via ROS 2 launch file (single container)
###############################################################################
set -euo pipefail
cd "$(dirname "$0")"

IMAGE_NAME="mcl_irpp_project:latest"

# Allow local X11 connections (for RViz / matplotlib)
xhost +local:docker 2>/dev/null || true

case "${1:-all}" in

  build)
    echo "▶ Building Docker image …"
    docker compose build
    ;;

  up)
    echo "▶ Starting containers …"
    docker compose up
    ;;

  down)
    echo "▶ Stopping containers …"
    docker compose down
    ;;

  shell)
    echo "▶ Opening interactive shell …"
    docker run --rm -it \
      --network host \
      -e DISPLAY="$DISPLAY" \
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      "$IMAGE_NAME" bash
    ;;

  launch)
    echo "▶ Running via ROS 2 launch file (single container) …"
    docker run --rm -it \
      --network host \
      -e DISPLAY="$DISPLAY" \
      -e QT_X11_NO_MITSHM=1 \
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      "$IMAGE_NAME" \
      bash -c "source /opt/ros/humble/setup.bash && \
               source /home/ubuntu/mcl_ws/install/setup.bash && \
               ros2 launch /home/ubuntu/mcl_ws/launch/mcl_launch.py"
    ;;

  all|"")
    echo "▶ Building Docker image …"
    docker compose build
    echo ""
    echo "▶ Starting all services …"
    docker compose up
    ;;

  *)
    echo "Usage: $0 {build|up|down|shell|launch|all}"
    exit 1
    ;;
esac
