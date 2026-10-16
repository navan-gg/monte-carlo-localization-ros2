 # Monte Carlo Localization (Particle Filter) in ROS2

This project was made as a part of the course EC4.202 Introduction to Robotics: Perception and Planning taken in Spring '25 in IIIT Hyderabad. The full write-up, including the maths, results and the effect of particle count on accuracy, is in [report/MCL_Project_Report.pdf](report/MCL_Project_Report.pdf).


A 2D particle filter (Monte Carlo Localization) written from scratch as a ROS 2 Humble node. It localizes a simulated robot in a grid world using range-bearing measurements to known landmarks, and is compared side by side against the Nav2 AMCL package on the same map and robot. The robot starts with no knowledge of its position, so particles are spread over the whole map. As the robot moves and senses landmarks, the particles converge on its true position. Every 30 seconds the robot is teleported ("kidnapped") to a random spot , and the filter has to find it again.

## How it works

**Custom MCL node** (`src/mcl_localization`)

- **Predict:** odometry motion model (rot1, trans, rot2) with Gaussian noise added to each particle
- **Update:** each particle is weighted by the Gaussian likelihood of the observed range and bearing to every visible landmark
- **Resample:** low-variance (systematic) resampling
- **Kidnap recovery:** 15% of particles are replaced with random ones at every resampling step, and if all weights stay very low for several updates in a row, the filter declares the robot lost and spreads all particles over the map again
- **Pose estimate:** weighted mean of the particles

**Simulated robot** (`src/mcl_robot`)

- Drives around the map on its own and avoids running into walls or corners
- Senses 5 landmarks within 8 m, with simulated noise on range and bearing of the landmark
- Is kidnapped to a random free position every 30 seconds

**Comparison with Nav2 AMCL** (`bridge/`)

AMCL needs a laser scan and odometry instead of landmarks. The scan bridge creates both from the simulated robot: a 36-beam laser scan by ray-casting on the map, and odometry from the robot's motion. When it detects a kidnap it asks AMCL to relocalize globally, so both filters face the same problem. AMCL then spreads up to 20,000 particles over the map and its adaptive (KLD) sampling reduces the count once it has converged.

## Results

Measured over 20 kidnaps per map (two 330 s runs). Full details are in the report.

| | `map` | `map2` | `map4` |
|---|---|---|---|
| Custom MCL: kidnaps recovered | 17 / 20 | 15 / 20 | 19 / 20 |
| Custom MCL: median error after recovery | 0.29 m | 0.40 m | 0.25 m |
| Nav2 AMCL: kidnaps recovered | 11 / 20 | 5 / 20 | 19 / 20 |
| Nav2 AMCL: median error after recovery | 0.06 m | 0.06 m | 0.06 m |

The custom MCL recovers from a kidnap faster (median 0.7–2.8 s, against 5.4–16.3 s for AMCL), but AMCL then tracks the robot more precisely. On the symmetric maps almost every AMCL failure was the same: it settled on a rotated copy of true poste. This isn't a problem in real world scenarios as maps are almost never symmetrical.

## Project structure

```
src/
  mcl_localization/      Custom MCL node (mcl_node.py)
  mcl_robot/             Simulated robot (simple_robot.py)
bridge/
  scan_bridge.py         Laser scan, odometry and TF for AMCL
  amcl_particle_viz.py   Thinned-out AMCL particle cloud for RViz
  amcl_params.yaml       Nav2 AMCL configuration
plots/
  mcl_plots.py           Live plot: custom MCL error and confidence
  comparison_plot.py     Live plot: custom MCL vs AMCL error, with statistics
  results_logger.py      Saves errors to CSV and a summary plot to results/
maps/                    Occupancy grid maps (map, map2, map3, map4) and the scripts that generate map2, map3 and map4
launch/mcl_launch.py     Single-container launch (custom MCL only, no AMCL)
rviz/mcl.rviz            RViz configuration
report/
  MCL_Project_Report.pdf Project report
  report.tex             LaTeX source of the report
  figures/               Figures used in the report
docker/                  Container entrypoint and DDS settings
Dockerfile
docker-compose.yml       Runs each part in its own container
run.sh                   Helper script for building and running
results/                 Output of the results logger (created when you run it)
```

## Requirements

- Linux with an X11 desktop (tested on Ubuntu 22.04 and 24.04)
- Docker with Docker Compose v2

Everything else (ROS 2 Humble, Nav2, RViz, Python packages) is installed inside the Docker image.

## How to run

Allow Docker containers to open windows on your screen:

```
xhost +local:docker
```

Build the image (the first build downloads ROS 2 and takes several minutes, subsequent builds run faster):

```
docker compose build
```

Start everything:

```
docker compose up
```

After about 30 seconds three windows open:

- **RViz:** the map, landmarks (yellow), the robot and its path, custom MCL particles (red), AMCL particles (purple), and the true (green), custom MCL (blue) and AMCL (yellow) pose arrows
- **Custom MCL:** custom MCL error and confidence over time
- **Custom MCL vs Nav2 AMCL:** error of both filters over time, with kidnap events marked and a table of running statistics

To stop, press `Ctrl+C` or run:

```
docker compose down
```

`run.sh` wraps these steps: `./run.sh` builds and starts everything, and `./run.sh down` stops it.

### Choosing a map

Four maps are included:

| Map | Layout |
|-----|--------|
| `map4` (default) | `map` with two extra obstacles so that it has no symmetry |
| `map` | three blocks, symmetric under a 180 degree rotation |
| `map2` | five hollow squares in a plus shape, symmetric under 90 degree rotations |
| `map3` | four identical closed rooms |

To use a different one:

```
MAP=map2 docker compose up
```

On the symmetric maps, AMCL often recovers from a kidnap to the wrong pose. On `map`, for example, a laser scan taken at (x, y) is identical to one taken at (-x, -y), so AMCL can settle on the mirrored pose and track it accurately while being far from the robot. The custom MCL does not have this problem because it knows which landmark each measurement comes from. This is why `map4` is the default.

### Saved results

While it runs, the results logger saves the following to `results/` every 10 seconds and once more when stopped:



