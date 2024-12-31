import genesis as gs
import torch
import numpy as np
import math
from genesis.engine.entities.drone_entity import DroneEntity
from genesis.vis.camera import Camera

base_rpm = 14468.429183500699
max_rpm = 25000
delta_rpm = 1000

def hover(drone: DroneEntity):
    drone.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])

def clamp(rpm):
    return max(0, min(int(rpm), max_rpm))

def fly_to_point(point, drone: DroneEntity, scene: gs.Scene, cam: Camera):
    pos = drone.get_pos()
    x_err = point[0]-pos[0]
    y_err = point[1]-pos[1]
    z_err = point[2]-pos[2]
    step = 0

    distance_to_point  = math.sqrt(x_err**2 + y_err**2 + z_err**2)

    while distance_to_point >= 0.1 or step < 500:
        rpm_offset_x = delta_rpm * x_err
        rpm_offset_y = delta_rpm * y_err
        rpm_offset_z = delta_rpm * z_err

        rpm_front_left  = clamp(base_rpm + (rpm_offset_x - rpm_offset_y + rpm_offset_z))
        rpm_front_right = clamp(base_rpm + (rpm_offset_x + rpm_offset_y + rpm_offset_z))
        rpm_rear_left   = clamp(base_rpm + (-rpm_offset_x - rpm_offset_y + rpm_offset_z))
        rpm_rear_right  = clamp(base_rpm + (-rpm_offset_x + rpm_offset_y + rpm_offset_z))
        
        rpms = torch.asarray([rpm_rear_right, rpm_front_right, rpm_front_left, rpm_rear_left])

        cpu_rpms = rpms.cpu()

        drone.set_propellels_rpm(cpu_rpms.numpy())

        scene.step()
        cam.render()
        print("point = ", drone.get_pos())
        drone_pos = drone.get_pos()
        drone_pos = drone_pos.cpu().numpy()
        cam.set_pose(lookat=(drone_pos[0], drone_pos[1], drone_pos[2]))
        x_err = point[0]-drone_pos[0]
        y_err = point[1]-drone_pos[1]
        z_err = point[2]-drone_pos[2]
        distance_to_point  = math.sqrt(x_err**2 + y_err**2 + z_err**2)
        step += 1

    # for i in range(250):
    #     drone.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])
    #     scene.step()
    #     cam.render()
    #     print("point = ", drone.get_pos())
    #     drone_pos = drone.get_pos()
    #     drone_pos = drone_pos.cpu().numpy()
    #     cam.set_pose(lookat=(drone_pos[0], drone_pos[1], drone_pos[2]))

def main():
    gs.init(backend=gs.gpu)
    
    ##### scene #####
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0,0,-9.81)
        )

    )

    ##### entities #####
    plane = scene.add_entity(
        morph=gs.morphs.Plane()
    )

    drone =  scene.add_entity(
        morph = gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0,0,0.2)
        )
    )

    cam = scene.add_camera(
        pos=(1,1,1),
        lookat=drone.morph.pos,
        GUI=False,
        res = (640,480),
        fov=30
    )

    ##### build #####

    scene.build()

    cam.start_recording()

    points = [
        (0,0,0.5)
    ]

    for point in points:
        fly_to_point(point, drone, scene, cam)

    cam.stop_recording(save_to_filename="./videos/fly_route.mp4")

if __name__ == "__main__":
    main()