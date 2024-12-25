import genesis as gs
import torch
import numpy as np
import math
from genesis.engine.entities.drone_entity import DroneEntity

base_rpm = 14468.429183500699
delta_rpm = 200

def hover(drone: DroneEntity):
    drone.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])

def fly_to_point(point, drone: DroneEntity):
    (x_pos, y_pos, z_pos) = drone.morph.pos
    x_err = x_pos-point[0]
    y_err = y_pos-point[1]
    z_err = z_pos-point[2]

    distance_to_point  = math.sqrt(x_err**2 + y_err**2 + z_err**2)

    while distance_to_point >= 1:
        rpm_offset_x = delta_rpm * distance_to_point * x_err
        rpm_offset_y = delta_rpm * distance_to_point * y_err
        rpm_offset_z = delta_rpm * distance_to_point * z_err

        rpm_front_left  = base_rpm + (rpm_offset_x - rpm_offset_y + rpm_offset_z)
        rpm_front_right = base_rpm + (rpm_offset_x + rpm_offset_y + rpm_offset_z)
        rpm_rear_left   = base_rpm + (-rpm_offset_x - rpm_offset_y + rpm_offset_z)
        rpm_rear_right  = base_rpm + (-rpm_offset_x + rpm_offset_y + rpm_offset_z)

        drone.set_propellels_rpm([rpm_front_left, rpm_front_right, rpm_rear_left, rpm_rear_right])

