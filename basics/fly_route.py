import genesis as gs
import torch
import numpy as np
import math
from genesis.engine.entities.drone_entity import DroneEntity
from genesis.vis.camera import Camera
from genesis.utils.geom import quat_to_xyz, quat_to_R, transform_by_quat, transform_quat_by_quat, inv_quat

base_rpm = 14468.429183500699
min_rpm = 0.98 * base_rpm
max_rpm = 1.5 * base_rpm
delta_rpm = 50

def hover(drone: DroneEntity):
    drone.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])

def clamp(rpm):
    return max(min_rpm, min(int(rpm), max_rpm))

def calc_dir_err(target, drone: DroneEntity):
    quat = drone.get_quat()
    pos = drone.get_pos()
    end = torch.Tensor([target[0]-pos[0], target[1]-pos[1], target[2]-pos[2]])
    dir = torch.mm(quat_to_R(quat).to("cpu"), torch.Tensor([[0],[0],[1]], device="cpu"))
    end = torch.nn.functional.normalize(end, dim=0)
    dir = torch.nn.functional.normalize(dir, dim=0)
    print("end =", end, "dir =", dir)
    return torch.norm(end.sub(dir))

def tilt_in_bounds(drone: DroneEntity):
    quat = drone.get_quat()
    euler = quat_to_xyz(quat)
    for angle in euler:
        if angle >= 10:
            print("out of bounds!")
            return False
    return True
    
def fly_to_point(point, drone: DroneEntity, scene: gs.Scene, cam: Camera):
    drone_pos = drone.get_pos()
    x_err = point[0]-drone_pos[0]
    y_err = point[1]-drone_pos[1]
    z_err = point[2]-drone_pos[2]
    step = 0

    while calc_dir_err(point, drone) >= 0.5 and step < 100:
        if not tilt_in_bounds(drone):
            break

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
        print("point =", drone.get_pos())
        drone_pos = drone.get_pos()
        drone_pos = drone_pos.cpu().numpy()
        cam.set_pose(lookat=(drone_pos[0], drone_pos[1], drone_pos[2]))
        step += 1

    step = 0
    distance_to_point = math.inf

    while distance_to_point > 0.25 and step < 250:
        drone.set_propellels_rpm([1.25*base_rpm, 1.25*base_rpm, 1.25*base_rpm, 1.25*base_rpm])
        scene.step()
        cam.render()
        print("point =", drone.get_pos())
        drone_pos = drone.get_pos()
        drone_pos = drone_pos.cpu().numpy()
        cam.set_pose(lookat=(drone_pos[0], drone_pos[1], drone_pos[2]))
        x_err = point[0]-drone_pos[0]
        y_err = point[1]-drone_pos[1]
        z_err = point[2]-drone_pos[2]
        distance_to_point  = math.sqrt(x_err**2 + y_err**2 + z_err**2)
        step += 1

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
        (1,1,2)
    ]

    for point in points:
        fly_to_point(point, drone, scene, cam)

    cam.stop_recording(save_to_filename="./videos/fly_route.mp4")

if __name__ == "__main__":
    main()