"""Passive controller for robot bodies moved by the supervisor.

The example focuses on formal multi-agent decision making.  Physical robot
poses are updated by the Webots Supervisor so that students can inspect the
auction and game-theoretic code without motor-control boilerplate.
"""

from controller import Robot


robot = Robot()
time_step = int(robot.getBasicTimeStep())

while robot.step(time_step) != -1:
    pass
