import genesis as gs
import torch
import numpy as np
import math
from genesis.engine.entities.drone_entity import DroneEntity
from genesis.vis.camera import Camera
import quaternion as qt

class QuatMath:

    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions: q1 * q2
        Quaternion format: [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
        ])
    
    def conjugate(q: np.ndarray) -> np.ndarray:
        """Return conjugate of quaternion [w, x, y, z] -> [w, -x, -y, -z]"""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def rotate_vector(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Rotate a vector v by quaternion q
        v_rotated = q * v * q_conjugate
        """
        # Convert vector to quaternion format [0, x, y, z]
        v_quat = np.array([0, v[0], v[1], v[2]])
        
        # Perform rotation
        q_conj = QuatMath.conjugate(q)
        v_rotated = QuatMath.multiply(
            QuatMath.multiply(q, v_quat),
            q_conj
        )
        
        # Return vector part
        return v_rotated[1:]

    def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle representation to quaternion"""
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2
        sin_half = np.sin(half_angle)
        return np.array([
            np.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        ])
    
    def get_yaw_from_quaternion(q: np.ndarray) -> float:
        """
        Extract yaw angle from quaternion.
        Returns angle in radians.
        """
        # Convert quaternion to rotation matrix
        R = np.array([
            [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
            [2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
            [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 1 - 2*q[1]**2 - 2*q[2]**2]
        ])
        
        # Extract yaw from rotation matrix
        return np.arctan2(R[1,0], R[0,0])

base_rpm = 14468.429183500699
thrust_rpm = 14700
max_rpm = 25000
delta_rpm = 200
delta_rotate = 100

def hover(drone: DroneEntity):
    drone.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])

def clamp(rpm):
    return max(0, min(int(rpm), max_rpm))

def calc_rpms(target, drone: DroneEntity) -> np.ndarray:
    pos = drone.get_pos()
    x_err = target[0]-pos[0]
    y_err = target[1]-pos[1]
    z_err = target[2]-pos[2]

    dir_vec = torch.asarray([x_err, y_err, z_err]).cpu().numpy()

    to_acc = delta_rpm * dir_vec

    thrust_vec = to_acc / np.linalg.norm(to_acc)

    # print(drone.get_quat().cpu().numpy())

    thrust_curr = QuatMath.rotate_vector(q=drone.get_quat().cpu().numpy(), v=np.array([0.,0.,1.]))

    rotation_axis = np.cross(thrust_curr, thrust_vec)
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(np.clip(
            np.dot(thrust_curr, thrust_vec), -1.0, 1.0
        ))
        
        desired_rotation = QuatMath.from_axis_angle(
            rotation_axis,
            rotation_angle * delta_rotate
        )
    else:
        desired_rotation = np.array([1., 0., 0., 0.])

    current_yaw = QuatMath.get_yaw_from_quaternion(drone.get_quat().cpu().numpy())
        
    # Calculate yaw error and corresponding rotation
    yaw_error = -current_yaw
    # Normalize to [-pi, pi]
    yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
    yaw_rotation = QuatMath.from_axis_angle(
        np.array([0., 0., 1.]),
        yaw_error * 2.5
    )
    
    # Combine thrust and yaw rotations
    desired_rotation = QuatMath.multiply(yaw_rotation, desired_rotation)

    quat_error = QuatMath.multiply(desired_rotation, drone.get_quat().cpu().numpy().conjugate())

    roll_adjustment = quat_error[0] * delta_rpm
    pitch_adjustment = quat_error[1] * delta_rpm
    yaw_adjustment = quat_error[2] * delta_rpm
    
    rpms = np.array([
        thrust_rpm - pitch_adjustment - roll_adjustment - yaw_adjustment,  # Motor 0
        thrust_rpm - pitch_adjustment + roll_adjustment + yaw_adjustment,  # Motor 1
        thrust_rpm + pitch_adjustment - roll_adjustment + yaw_adjustment,  # Motor 2
        thrust_rpm + pitch_adjustment + roll_adjustment - yaw_adjustment   # Motor 3
    ])

    return rpms

def fly_to_point(target, drone: DroneEntity, scene: gs.Scene, cam: Camera):
    pos = drone.get_pos()
    x_err = target[0]-pos[0]
    y_err = target[1]-pos[1]
    z_err = target[2]-pos[2]
    steps = 0

    distance_to_point  = math.sqrt(x_err**2 + y_err**2 + z_err**2)

    # while distance_to_point >= 1 and steps < 60:

    #     rpms = calc_rpms(target, drone)

    #     drone.set_propellels_rpm(rpms)

    #     scene.step()
    #     cam.render()
    #     print("point = ", drone.get_pos())
    #     drone_pos = drone.get_pos()
    #     drone.get_quat()
    #     drone_pos = drone_pos.cpu().numpy()
    #     cam.set_pose(lookat=(drone_pos[0], drone_pos[1], drone_pos[2]))

    #     steps += 1

    for i in range(250):
        drone.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])
        scene.step()
        cam.render()

def main():
    gs.init(backend=gs.gpu)
    
    ##### scene #####
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(
            dt=0.01
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
        (5,3,2)
    ]

    for point in points:
        fly_to_point(point, drone, scene, cam)

    cam.stop_recording(save_to_filename="./videos/fly_route.mp4")

if __name__ == "__main__":
    main()