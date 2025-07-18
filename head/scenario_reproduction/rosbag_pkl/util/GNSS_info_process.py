#! /usr/bin/env python3


import socket
import struct
import numpy as np


class GNSS_UDP_receive:
    def __init__(self):
        self.OutputRecordType = None
        self.RecordLength = None
        self.GPSWeek = None
        self.GPSTime = None
        self.IMUAlignmentStatus = None
        self.GNSSStatus = None
        self.Latitude = None
        self.Longitude = None
        self.Altitude = None
        self.North_Velocity = None
        self.East_Velocity = None
        self.Velocitydown = None
        self.TotalVelocity = None
        self.Roll = None
        self.Pitch = None
        self.Heading = None
        self.TrackingAngle = None
        self.RollRate = None
        self.PitchRate = None
        self.Yawrate = None
        self.ax = None
        self.ay = None
        self.az = None
        self.recv_msg = None

    def GNSS_resolution(self):
        # 按照惯导协议解析数据
        self.UDP_receive()
        STX02h_recv = self.recv_msg[0:1]
        Status_recv = self.recv_msg[1:2]
        Packettype40h_recv = self.recv_msg[2:3]
        Length_recv = self.recv_msg[3:4]
        TransNum_recv = self.recv_msg[4:5]
        PageIndex_recv = self.recv_msg[5:6]
        Maxpageindex_recv = self.recv_msg[6:7]
        OutputRecordType_recv = self.recv_msg[7:8]
        RecordLength_recv = self.recv_msg[8:9]
        GPSWeek_recv = self.recv_msg[9:11]
        GPSTime_recv = self.recv_msg[11:15]
        IMUAlignmentStatus_recv = self.recv_msg[15:16]
        GNSSStatus_recv = self.recv_msg[16:17]
        Latitude_recv = self.recv_msg[17:25]
        Longitude_recv = self.recv_msg[25:33]
        Altitude_recv = self.recv_msg[33:41]
        North_Velocity_recv = self.recv_msg[41:45]
        East_Velocity_recv = self.recv_msg[45:49]
        Velocitydown_recv = self.recv_msg[49:53]
        TotalVelocity_recv = self.recv_msg[53:57]
        Roll_recv = self.recv_msg[57:65]
        Pitch_recv = self.recv_msg[65:73]
        Heading_recv = self.recv_msg[73:81]
        TrackingAngle_recv = self.recv_msg[81:89]
        Rollrate_recv = self.recv_msg[89:93]
        Pitchrate_recv = self.recv_msg[93:97]
        Yawrate_recv = self.recv_msg[97:101]
        ax_recv = self.recv_msg[101:105]
        ay_recv = self.recv_msg[105:109]
        az_recv = self.recv_msg[109:113]

        self.OutputRecordType = struct.unpack('>?', OutputRecordType_recv)[0]
        self.RecordLength = struct.unpack('>?', RecordLength_recv)[0]
        self.GPSWeek = struct.unpack('>h', GPSWeek_recv)[0]
        self.GPSTime = struct.unpack('>L', GPSTime_recv)[0]
        self.IMUAlignmentStatus = struct.unpack('>?', IMUAlignmentStatus_recv)[0]
        self.GNSSStatus = struct.unpack('>?', GNSSStatus_recv)[0]
        self.Latitude = struct.unpack('>d', Latitude_recv)[0]
        self.Longitude = struct.unpack('>d', Longitude_recv)[0]
        self.Altitude = struct.unpack('>d', Altitude_recv)[0]
        self.North_Velocity = struct.unpack('>f', North_Velocity_recv)[0]
        self.East_Velocity = struct.unpack('>f', East_Velocity_recv)[0]
        self.Velocitydown = struct.unpack('>f', Velocitydown_recv)[0]
        self.TotalVelocity = struct.unpack('>f', TotalVelocity_recv)[0]
        self.Roll = struct.unpack('>d', Roll_recv)[0]
        self.Pitch = struct.unpack('>d', Pitch_recv)[0]
        self.Heading = struct.unpack('>d', Heading_recv)[0]
        self.TrackingAngle = struct.unpack('>d', TrackingAngle_recv)[0]
        self.RollRate = struct.unpack('>f', Rollrate_recv)[0]
        self.PitchRate = struct.unpack('>f', Pitchrate_recv)[0]
        self.Yawrate = struct.unpack('>f', Yawrate_recv)[0]
        self.ax = struct.unpack('>f', ax_recv)[0]
        self.ay = struct.unpack('>f', ay_recv)[0]
        self.az = struct.unpack('>f', az_recv)[0]

    def UDP_receive(self):
        # 1创建套接字
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 2.绑定一个本地信息
        localaddr = ("", 8234)  # 必须绑定自己电脑IP和port
        udp_socket.bind(localaddr)

        # 3.接收数据
        recv_data = udp_socket.recvfrom(115)
        self.recv_msg = recv_data[0]  # 信息内容


class GPS_to_Global:
    def __init__(self):
        self.Latitude = None
        self.Longitude = None
        self.a = 6378137.00000000000000
        self.f = 1 / 298.257223563  # 地球的扁率
        self.b = self.a * (1.0 - self.f)  # 地球的短半轴
        self.e = np.sqrt((self.a ** 2.0 - self.b ** 2.0) / self.a ** 2.0)  # 第一偏心率
        self.ep = np.sqrt((self.a ** 2.0 - self.b ** 2.0) / self.b ** 2.0)  # 第

    def GPS(self, lat, lon):  # WGS48是椭球参数，a是地球长半轴

        LAT_rad = lat * np.pi / 180.0  # 改角度为弧度

        LON_rad = lon * np.pi / 180.0

        Num = int((LON_rad * 180.0 / np.pi) / 3.0)
        L0 = 3 * Num * np.pi / 180.0  # L0 中央子午线经度  rad
        l = LON_rad - L0  # 经度差
        N = self.a / np.sqrt(1.0 - self.e ** 2.0 * np.sin(LAT_rad) ** 2.0)  # N 卯酉圈的半径
        t = np.tan(LAT_rad)
        eta = self.ep * np.cos(LAT_rad)

        m0 = self.a * (1 - self.e ** 2.0)
        m2 = 3.0 / 2.0 * self.e ** 2.0 * m0
        m4 = 5.0 / 4.0 * self.e ** 2.0 * m2
        m6 = 7.0 / 6.0 * self.e ** 2.0 * m4
        m8 = 9.0 / 8.0 * self.e ** 2.0 * m6

        a0 = m0 + m2 / 2.0 + 3.0 / 8 * m4 + 5.0 / 16 * m6 + 35.0 / 128 * m8
        a2 = m2 / 2.0 + m4 / 2.0 + 15.0 / 32 * m6 + 7.0 / 16 * m8
        a4 = m4 / 8.0 + 3.0 / 16 * m6 + 7.0 / 32 * m8
        a6 = m6 / 32.0 + m8 / 16.0
        a8 = m8 / 128.0
        Y = a0 * LAT_rad - (a2 * np.sin(2.0 * LAT_rad)) / 2.0 + (a4 * np.sin(4.0 * LAT_rad)) / 4 - (
                a6 * np.sin(6 * LAT_rad)) / 6 + (
                    a8 * np.sin(8 * LAT_rad)) / 8
        # 接下来我们计算x,y
        y = Y + N / 2.0 * np.sin(LAT_rad) * np.cos(LAT_rad) * l ** 2 + N / 24.0 * np.sin(LAT_rad) * (
                np.cos(LAT_rad) ** 3) * (
                    5.0 - (t ** 2.0) + 9 * eta ** 2 + 4.0 * eta ** 4) * (l ** 4) + N / 720 * np.sin(LAT_rad) * (
                    np.cos(LAT_rad) ** 5) * (
                    61.0 - 58.0 * (t ** 2) + t ** 4 + 270.0 * (eta ** 2) - 330.0 * (eta ** 2) * (t ** 2)) * (l ** 6)
        x = N * np.cos(LAT_rad) * l + N / 6.0 * (np.cos(LAT_rad) ** 3) * (1 - t ** 2 + eta ** 2.0) * (
                l ** 3) + N / 120.0 * (
                    np.cos(LAT_rad) ** 5) * (
                    5.0 - 18 * (t ** 2) + t ** 4 + 14.0 * (eta ** 2) - 58.0 * (eta ** 2) * (t ** 2)) * (
                    l ** 5)
        x = x + 500000

        # global_x = abs(x - 615000)
        # global_y = abs(y - 3463000)

        global_x = x - 615000
        global_y = y - 3463000
        return global_x, global_y
