import numpy as np

def glorot_uniform(fan_out,fan_in):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_out, fan_in))