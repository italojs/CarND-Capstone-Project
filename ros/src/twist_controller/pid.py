
MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.beta = 0.0   # integrate on-off
        self.min = mn
        self.max = mx

        self.int_val = self.last_error = 0.

    def reset(self):
        self.int_val = 0.0

    def step(self, error, sample_time):
        # if error too large,
        integral = 0.0
        if error < 6.0 and error > -6.0:
            self.beta = 1.0
            integral = self.int_val + error * sample_time;
        derivative = (error - self.last_error) / sample_time;

        # restrain the too large varie in short time
        # val = self.kp * error + self.ki * integral + self.kd  * derivative
        val = self.kp * error + self.beta * self.ki * integral + self.kd *(derivative/(1 + derivative)) * derivative
        if val > self.max:
            val = self.max
        elif val < self.min:
            val = self.min
        else:
            self.int_val = integral
        self.last_error = error

        return val
