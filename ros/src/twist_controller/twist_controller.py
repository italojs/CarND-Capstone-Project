import rospy

from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, deceleration_limit, acceleration_limit, wheel_radius, wheel_base, steer_ratio,
                 max_lat_accel, max_steer_angle, acc_kp, acc_ki, acc_kd):
        # Init lateral controller         
        self.yaw_controller = YawController(wheel_base=wheel_base, steer_ratio=steer_ratio, min_speed=0.1,
                                            max_lat_accel=max_lat_accel, max_steer_angle=max_steer_angle)

        # Init longitudinal controller
        mn = 0.0
        mx = 0.2
        self.throttle_controller = PID(acc_kp, acc_ki, acc_kd, mn, mx)

        tau = 0.5
        ts = 0.02
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.deceleration_limit = deceleration_limit
        self.acceleration_limit = acceleration_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()
        self.last_vel = 0.0

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0
        self.last_vel = current_vel

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        # Write to ros log
        # rospy.logwarn("Angular vel: {0}".format(angular_vel))
        rospy.logwarn("Vref, Vfb:[%s, %s][m/s]",format(linear_vel),format(current_vel))
        # rospy.logwarn("Target angular velocity: {0}".format(angular_vel))
        # rospy.logwarn("Current velocity: {0}".format(current_vel))
        # rospy.logwarn("Filtered velocity: {0}".format(self.vel_lpf.get()))
   
        vel_error = linear_vel - current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        if linear_vel == 0.0 and current_vel < 0.1:
            throttle = 0.0
            brake = 700
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            deceleration = max(vel_error, self.deceleration_limit)
            brake = abs(deceleration) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steering
