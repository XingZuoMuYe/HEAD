import math
import numpy as np


class GNSS_Transform:
    def __init__(self):
        # 地图匹配坐标变换变量_西区
        self.theta_g2c_west = 0.8 + 180.0
        self.theta_c2g_west = -self.theta_g2c_west
        self.dx_west = 169.2974
        self.dy_west = 846.6133
        # 地图匹配坐标变换变量_东区
        self.theta_g2c_east = -0.55
        self.theta_c2g_east = -self.theta_g2c_east
        self.dx_east = 182.3985
        self.dy_east = -844.3248

    def GNSS_Global_to_CarlaEast(self, Global_X, Global_Y, Global_Yaw):
        '''由惯导坐标转carla坐标，更新位资信息'''
        '''东区坐标转换是先位移再旋转'''
        theta = self.theta_g2c_east * math.pi / 180.0
        x_temp = Global_X + self.dx_east
        y_temp = Global_Y + self.dy_east

        x = math.cos(theta) * x_temp - math.sin(theta) * y_temp  # 2.3 2.4
        y = math.sin(theta) * x_temp + math.cos(theta) * y_temp

        Carla_X = x
        Carla_Y = -y
        Carla_Yaw = Global_Yaw - 90.0

        return Carla_X, Carla_Y, Carla_Yaw

    def GNSS_CarlaEast_to_Global(self, Carla_X, Carla_Y, Carla_Yaw):
        '''将Carla坐标系的点转换到世界坐标系'''
        theta = self.theta_c2g_east * math.pi / 180.0
        y_trans = -Carla_Y
        Global_X = math.cos(theta) * Carla_X - math.sin(theta) * y_trans - self.dx_east
        Global_Y = math.sin(theta) * Carla_X + math.cos(theta) * y_trans - self.dy_east
        Global_Yaw = Carla_Yaw + 90.0

        return Global_X, Global_Y, Global_Yaw

    def GNSS_Global_to_CarlaWest(self, Global_X, Global_Y, Global_Yaw):
        '''由惯导坐标转carla坐标，更新位资信息'''
        '''西区坐标转换是先旋转再位移'''
        # 西区地图匹配时，把GPS转Global的global_x,global_y取了绝对值abs，去掉绝对值需要将Global_X取个负数
        Global_X = -Global_X

        theta = self.theta_g2c_west * math.pi / 180.0
        x_tmp = math.cos(theta) * Global_X - math.sin(theta) * Global_Y  # 2.3 2.4
        y_tmp = math.sin(theta) * Global_X + math.cos(theta) * Global_Y
        Carla_X = x_tmp + self.dx_west
        Carla_Y = y_tmp + self.dy_west
        # Global_Yaw = 180.0 - Global_Yaw
        # Carla_Yaw = 90.8 - Global_Yaw
        Carla_Yaw = Global_Yaw - 89.2

        return Carla_X, -Carla_Y, Carla_Yaw  # 为解决右舵驾驶问题地图关于x轴镜像，Carla_Y取负

    def GNSS_CarlaWest_to_Global(self, Carla_X, Carla_Y, Carla_Yaw):
        '''将Carla坐标系的点转换到世界坐标系'''
        theta = self.theta_c2g_west * math.pi / 180.0
        x_temp = Carla_X - self.dx_west
        y_temp = Carla_Y - self.dy_west
        Global_X = x_temp * math.cos(theta) - y_temp * math.sin(theta)
        Global_Y = x_temp * math.sin(theta) + y_temp * math.cos(theta)
        # Global_Yaw = 90.8 - Carla_Yaw
        # Global_Yaw = 180.0 - Global_Yaw
        Global_Yaw = 89.2 + Carla_Yaw

        Global_X = -Global_X

        return Global_X, Global_Y, Global_Yaw

    def GNSS_Vehicle_to_Carla(self, Vehicle_X, Vehicle_Y, Vehicle_Carla_X, Vehicle_Carla_Y, Vehicle_Carla_Yaw):
        '''将自辆坐标系的点转换到Carla坐标系'''
        theta = Vehicle_Carla_Yaw * math.pi / 180.0
        x_temp = Vehicle_X * math.cos(theta) - Vehicle_Y * math.sin(theta)
        y_temp = Vehicle_X * math.sin(theta) + Vehicle_Y * math.cos(theta)
        Carla_X = x_temp + Vehicle_Carla_X
        Carla_Y = y_temp + Vehicle_Carla_Y

        return Carla_X, Carla_Y

    def GNSS_Carla_to_Vehicle(self, Carla_X, Carla_Y, Vehicle_Carla_X, Vehicle_Carla_Y, Vehicle_Carla_Yaw):
        '''将Carla坐标系的点转换到自辆坐标系'''
        theta = -Vehicle_Carla_Yaw * math.pi / 180.0
        x_temp = Carla_X - Vehicle_Carla_X
        y_temp = Carla_Y - Vehicle_Carla_Y
        Vehicle_X = x_temp * math.cos(theta) - y_temp * math.sin(theta)
        Vehicle_Y = x_temp * math.sin(theta) + y_temp * math.cos(theta)

        return Vehicle_X, Vehicle_Y

    def carla2global_east_vel(self, vx, vy):
        # carla坐标系转成东风车坐标系
        theta_c2g = 0.55
        theta = theta_c2g * math.pi / 180.0
        global_vx = vx * math.cos(theta) - vy * math.sin(theta)
        global_vy = vx * math.sin(theta) + vy * math.cos(theta)
        return global_vx, global_vy

    def carla2global_east_acc(self, vx, vy):
        # carla坐标系转成东风车坐标系
        theta_c2g = 0.55
        theta = theta_c2g * math.pi / 180.0
        global_vx = -vx * math.sin(theta) - vy * math.cos(theta)
        global_vy = vx * math.cos(theta) - vy * math.sin(theta)
        return global_vx, global_vy

    def Velocity_CarlaWest_to_Global(self, vx, vy):
        # carla坐标系转成东风车坐标系
        theta = self.theta_c2g_west * math.pi / 180.0
        Global_VX = vx * math.cos(theta) - vy * math.sin(theta)
        Global_VY = vx * math.sin(theta) + vy * math.cos(theta)

        return Global_VX, Global_VY

    def Acceleration_CarlaWest_to_Global(self, vx, vy):
        # carla坐标系转成东风车坐标系
        theta = self.theta_c2g_west * math.pi / 180.0
        global_vx = -vx * math.sin(theta) - vy * math.cos(theta)
        global_vy = vx * math.cos(theta) - vy * math.sin(theta)

        return global_vx, global_vy
