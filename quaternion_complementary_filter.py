# Implementing the quaternion-based complementary filter described in the article at https://www.mdpi.com/1424-8220/15/8/19302
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def quaternion_conjugate(q):
    """Conjugate of a quaternion."""
    q0 = q[0].item()
    q1 = q[1].item()
    q2 = q[2].item()
    q3 = q[3].item()
    return np.array([[q0], [-q1], [-q2], [-q3]])

def quaternion_multiply(p, q):
    """Multiply two quaternions."""
    p0 = p[0].item()
    p1 = p[1].item()
    p2 = p[2].item()
    p3 = p[3].item()
    q0 = q[0].item()
    q1 = q[1].item()
    q2 = q[2].item()
    q3 = q[3].item()
    return np.array([
        [p0*q0 - p1*q1 - p2*q2 - p3*q3],
        [p0*q1 + p1*q0 + p2*q3 - p3*q2],
        [p0*q2 - p1*q3 + p2*q0 + p3*q1],
        [p0*q3 + p1*q2 - p2*q1 + p3*q0],
    ])

def quaternion_to_euler(q):
    """Convert a quaternion into Euler angles."""
    w = q[0].item()
    x = q[1].item()
    y = q[2].item()
    z = q[3].item()

    # Compute yaw (z-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Compute pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Compute roll (x-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def quaternion_derivative(q, omega):
    """Calculate quaternion derivative from angular velocity."""
    return -0.5 * quaternion_multiply(np.append([0], omega), q)

def rotation_matrix_from_quaternion(q):
    """
    Convert a quaternion into a 3x3 rotation matrix.

    Parameters:
    q (array-like): Quaternion in the format [q0, q1, q2, q3]

    Returns:
    numpy.ndarray: 3x3 rotation matrix
    """
    q0 = q[0, 0].item()
    q1 = q[1, 0].item()
    q2 = q[2, 0].item()
    q3 = q[3, 0].item()

    return np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q3*q0, 2*q1*q3 + 2*q2*q0],
        [2*q1*q2 + 2*q3*q0, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q1*q0],
        [2*q1*q3 - 2*q2*q0, 2*q2*q3 + 2*q1*q0, 1 - 2*q1**2 - 2*q2**2]
    ])

def complementary_filter_step(q_prev, omega, accel, mag, dt, alpha, beta):
    """One step of the complementary filter."""
    # Prediction step
    q_dot = quaternion_derivative(q_prev, omega)
    q_pred = q_prev + q_dot * dt
    q_pred /= np.linalg.norm(q_pred)

    # Predicted gravity
    rot = rotation_matrix_from_quaternion(quaternion_conjugate(q_pred))
    g = rot @ accel
    gx = g[0, 0].item()
    gy = g[1, 0].item()
    gz = g[2, 0].item()

    # Delta acceleration quaternion
    delta_q_acc = np.array([
                            [np.sqrt((gz+1)/2)],
                            [-gy/(np.sqrt(2*(gz+1)))],
                            [gx/(np.sqrt(2*(gz+1)))],
                            [0]])
    
    # Interpolate with identity quaternion
    if delta_q_acc[0] > 0.9:
        # Linear Interpolation
        delta_q_acc_bar = (1.0 - alpha) * np.array([[1], [0], [0], [0]]) + alpha * delta_q_acc
        delta_q_acc_hat = normalize(delta_q_acc_bar)
    else:
        # Spherical Linear Interpolation
        omega = np.arccos(delta_q_acc[0])
        delta_q_acc_hat = (
            np.array([[1], [0], [0], [0]]) * np.sin((1.0 - alpha) * omega) / np.sin(omega) +
            delta_q_acc * np.sin(alpha * omega) / np.sin(omega)
        )

    # Prediction corrected by accelerometers
    q_pred_acc = quaternion_multiply(q_pred, delta_q_acc_hat)

    # Conjiugate quaternion
    q_pred_acc_conj = quaternion_conjugate(q_pred_acc)

    # Rotate magnetometer measurement
    lx = (rotation_matrix_from_quaternion(q_pred_acc_conj) @ mag)[0].item()
    ly = (rotation_matrix_from_quaternion(q_pred_acc_conj) @ mag)[1].item()
    lz = (rotation_matrix_from_quaternion(q_pred_acc_conj) @ mag)[2].item()

    # Delta magnetometer quaternion
    Gamma = np.sqrt(lx**2 + ly**2)
    delta_q_mag = np.array([
                            [np.sqrt(Gamma + lx*np.sqrt(Gamma)) / np.sqrt(2 * Gamma)],
                            [0],
                            [0],
                            [ly / (np.sqrt(2) * np.sqrt(Gamma + lx*np.sqrt(Gamma)))]])
    
    # Interpolate with identity quaternion
    if delta_q_mag[0] > 0.9:
        # Linear Interpolation
        delta_q_mag_bar = (1.0 - beta) * np.array([[1], [0], [0], [0]]) + beta * delta_q_mag
        delta_q_mag_hat = normalize(delta_q_mag_bar)
    else:
        # Spherical Linear Interpolation
        omega = np.arccos(delta_q_mag[0])
        delta_q_mag_hat = (
            np.array([[1], [0], [0], [0]]) * np.sin((1.0 - beta) * omega) / np.sin(omega) +
            delta_q_mag * np.sin(beta * omega) / np.sin(omega)
        )

    # Applying complementary filter
    q_next = quaternion_multiply(q_pred_acc, delta_q_mag_hat)

    return normalize(q_next)

# Configure filter
dt = 0.1  # Time step
alpha = 0.5  # Acc filter coefficient
beta = 0.5  # Mag filter coefficient

# Load the data
gyro_data = pd.read_csv('GyroscopeUncalibrated.csv')
mag_data = pd.read_csv('MagnetometerUncalibrated.csv')
accel_data = pd.read_csv('AccelerometerUncalibrated.csv')

# Resample the data to a uniform time axis
time_axis = np.arange(0, max(gyro_data['seconds_elapsed'].max(), mag_data['seconds_elapsed'].max(), accel_data['seconds_elapsed'].max()), dt)
gyro_data = gyro_data.set_index('seconds_elapsed').reindex(time_axis, method='nearest').reset_index()
mag_data = normalize(mag_data.set_index('seconds_elapsed').reindex(time_axis, method='nearest').reset_index())
accel_data = normalize(accel_data.set_index('seconds_elapsed').reindex(time_axis, method='nearest').reset_index())

# Initialize quaternion
q_prev = np.array([[1], [0], [0], [0]])

# Run the filter for each time step and log the quaternion
quaternion_log = []
pitch_log = []
roll_log = []
yaw_log = []
for i in range(len(time_axis)):
    omega = np.array([[gyro_data['x'].iloc[i]], [gyro_data['y'].iloc[i]], [gyro_data['z'].iloc[i]]])  # Angular velocity in rad/s from gyro data
    accel = np.array([[accel_data['x'].iloc[i]], [accel_data['y'].iloc[i]], [accel_data['z'].iloc[i]]])  # Accelerometer readings
    mag = np.array([[mag_data['x'].iloc[i]], [mag_data['y'].iloc[i]], [mag_data['z'].iloc[i]]])  # Magnetometer readings

    # Perform a single filter step
    q_next = complementary_filter_step(q_prev, omega, accel, mag, dt, alpha, beta)
    quaternion_log.append(q_next)
    q_prev = q_next

    # Conjugate
    q_next_conj = quaternion_conjugate(q_next)

    # Calculate pitch, roll, and yaw from the quaternion
    pitch, roll, yaw = quaternion_to_euler(q_next_conj)
    pitch_log.append(pitch)
    roll_log.append(roll)
    yaw_log.append(yaw)

# Convert quaternion log to a numpy array for easier manipulation
quaternion_log = np.array(quaternion_log)

# Create subplots for the quaternion and the pitch, roll, and yaw over time

# Plot the quaternion over time in the first subplot
plt.subplot(2, 1, 1)
plt.plot(time_axis, quaternion_log[:, 0], label='q0')
plt.plot(time_axis, quaternion_log[:, 1], label='q1')
plt.plot(time_axis, quaternion_log[:, 2], label='q2')
plt.plot(time_axis, quaternion_log[:, 3], label='q3')
plt.legend()
plt.title('Quaternion over time')

# Plot the pitch, roll and yaw over time in the second subplot
plt.subplot(2, 1, 2)
plt.plot(time_axis, pitch_log, label='Pitch')
plt.plot(time_axis, roll_log, label='Roll')
plt.plot(time_axis, yaw_log, label='Yaw')
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")

# Display the subplots
plt.show()

