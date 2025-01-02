import torch
import numpy as np
from genesis.engine.entities.drone_entity import DroneEntity
from genesis.utils.geom import quat_to_xyz, quat_to_R, transform_by_quat, transform_quat_by_quat, inv_quat

class PIDController():
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.
        self.prev_error = 0.

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        return (self.kp*error) + (self.ki*self.integral) + (self.kd*derivative)
    
class DroneController():
    def __init__(self, drone: DroneEntity, dt, base_rpm):
        self.__pid_pos_x = PIDController(kp=1., ki=0.0, kd=0.0)
        self.__pid_pos_y = PIDController(kp=1., ki=0.0, kd=0.0)
        self.__pid_pos_z = PIDController(kp=1., ki=0.0, kd=0.0)

        self.__pid_vel_x = PIDController(kp=10., ki=0.0, kd=0.)
        self.__pid_vel_y = PIDController(kp=10., ki=0.0, kd=0.)
        self.__pid_vel_z = PIDController(kp=50., ki=0.0, kd=0.)

        self.__pid_att_roll  = PIDController(kp=1., ki=0.0, kd=0.)
        self.__pid_att_pitch = PIDController(kp=1., ki=0.0, kd=0.)
        self.__pid_att_yaw   = PIDController(kp=1., ki=0.0, kd=0.)

        self.drone = drone
        self.__dt = dt
        self.__base_rpm = base_rpm
        self.__yaw_target = 0.

    def __get_drone_pos(self) -> torch.Tensor:
        return self.drone.get_pos()

    def __get_drone_vel(self) -> torch.Tensor:
        return self.drone.get_vel()
    
    def __get_drone_att(self) -> torch.Tensor:
        quat = self.drone.get_quat()
        # print(quat_to_xyz(quat))
        return quat_to_xyz(quat)
    
    def __mixer(self, thrust, roll, pitch, yaw) -> torch.Tensor:
        M1 = self.__base_rpm + (thrust - roll + pitch - yaw)
        M2 = self.__base_rpm + (thrust + roll + pitch + yaw)
        M3 = self.__base_rpm + (thrust + roll - pitch - yaw)
        M4 = self.__base_rpm + (thrust - roll - pitch + yaw)
        # print("pitch =", pitch)
        # print("roll =", roll)

        return torch.Tensor([M1, M2, M3, M4])

    def update(self, target) -> np.ndarray:
        curr_pos = self.__get_drone_pos()
        curr_vel = self.__get_drone_vel()
        curr_att = self.__get_drone_att()

        err_pos_x = target[0] - curr_pos[0]
        err_pos_y = target[1] - curr_pos[1]
        err_pos_z = target[2] - curr_pos[2]

        # print(err_pos_x)
        # print(err_pos_y)
        # print(err_pos_z)

        vel_des_x = self.__pid_pos_x.update(err_pos_x, self.__dt)
        vel_des_y = self.__pid_pos_y.update(err_pos_y, self.__dt)
        vel_des_z = self.__pid_pos_z.update(err_pos_z, self.__dt)

        error_vel_x = vel_des_x - curr_vel[0]
        error_vel_y = vel_des_y - curr_vel[1]
        error_vel_z = vel_des_z - curr_vel[2]

        # print(error_vel_x)
        # print(error_vel_y)
        # print(error_vel_z)

        roll_des   = self.__pid_vel_x.update(error_vel_x, self.__dt)
        pitch_des  = self.__pid_vel_y.update(error_vel_y, self.__dt)
        thrust_des = self.__pid_vel_z.update(error_vel_z, self.__dt)


        err_roll  = roll_des - curr_att[0]
        err_pitch = pitch_des - curr_att[1]
        err_yaw   = self.__yaw_target - curr_att[2]

        print(err_roll)
        print(err_pitch)
        print(err_yaw)

        roll_del  = self.__pid_att_roll.update(err_roll, self.__dt)
        pitch_del = self.__pid_att_pitch.update(err_pitch, self.__dt)
        yaw_del   = self.__pid_att_yaw.update(err_yaw, self.__dt)

        print(roll_del)
        print(pitch_del)
        print(yaw_del)
        print(thrust_des)

        prop_rpms = self.__mixer(thrust_des, roll_del, pitch_del, yaw_del)
        prop_rpms = prop_rpms.cpu()
        prop_rpms - prop_rpms.numpy()

        return prop_rpms