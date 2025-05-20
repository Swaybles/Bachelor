import math
import numpy as np
import matplotlib.pyplot as plt
import copy

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar, self.z * scalar)

class Body:
    def __init__(self, name, mass, location, velocity):
        self.name = name
        self.mass = mass
        self.location = location
        self.velocity = velocity

def gravitational_acceleration(satellite, earth):
    G = 6.67408e-11
    dx = earth.location.x - satellite.location.x
    dy = earth.location.y - satellite.location.y
    dz = earth.location.z - satellite.location.z
    r_sq = dx*dx + dy*dy + dz*dz
    r = math.sqrt(r_sq)
    if r == 0:
        return Point(0, 0, 0)
    acc = G * earth.mass / r_sq
    return Point(acc * dx / r, acc * dy / r, acc * dz / r)

def orbital_elements(body, mu):
    r_vec = np.array([body.location.x, body.location.y, body.location.z])
    v_vec = np.array([body.velocity.x, body.velocity.y, body.velocity.z])
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    i = math.acos(max(-1.0, min(1.0, h_vec[2] / h))) if h != 0 else 0
    k_vec = np.array([0, 0, 1])
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)
    e_vec = (1/mu) * (np.cross(v_vec, h_vec) - mu * (r_vec / r)) if r != 0 else np.zeros(3)
    e = np.linalg.norm(e_vec)
    a = 1 / ((2 / r) - (v**2 / mu)) if r != 0 else 0
    RAAN = 0
    if n != 0:
        cos_O = max(-1.0, min(1.0, n_vec[0] / n))
        RAAN = math.acos(cos_O)
        if n_vec[1] < 0:
            RAAN = 2*math.pi - RAAN
    omega = 0
    if n != 0 and e != 0:
        cos_w = max(-1.0, min(1.0, np.dot(n_vec, e_vec) / (n * e)))
        omega = math.acos(cos_w)
        if e_vec[2] < 0:
            omega = 2*math.pi - omega
    nu = 0
    if e != 0 and r != 0:
        cos_nu = max(-1.0, min(1.0, np.dot(e_vec, r_vec) / (e * r)))
        nu = math.acos(cos_nu)
        if np.dot(r_vec, v_vec) < 0:
            nu = 2*math.pi - nu
    return {"a": a, "e": e, "i": math.degrees(i),
            "RAAN": math.degrees(RAAN), "omega": math.degrees(omega), "nu": math.degrees(nu)}


def rk4_step(body, earth, dt):
    def deriv(loc, vel):
        accel = gravitational_acceleration(Body('', 0, loc, vel), earth)
        return vel, accel

    r0 = body.location
    v0 = body.velocity
    k1_r, k1_v = deriv(r0, v0)
    k2_r, k2_v = deriv(r0 + k1_r * (dt/2), Point(v0.x + k1_v.x*(dt/2), v0.y + k1_v.y*(dt/2), v0.z + k1_v.z*(dt/2)))
    k3_r, k3_v = deriv(r0 + k2_r * (dt/2), Point(v0.x + k2_v.x*(dt/2), v0.y + k2_v.y*(dt/2), v0.z + k2_v.z*(dt/2)))
    k4_r, k4_v = deriv(r0 + k3_r * dt,   Point(v0.x + k3_v.x*dt,     v0.y + k3_v.y*dt,     v0.z + k3_v.z*dt))
    
    new_loc = Point(
        r0.x + (dt/6)*(k1_r.x + 2*k2_r.x + 2*k3_r.x + k4_r.x),
        r0.y + (dt/6)*(k1_r.y + 2*k2_r.y + 2*k3_r.y + k4_r.y),
        r0.z + (dt/6)*(k1_r.z + 2*k2_r.z + 2*k3_r.z + k4_r.z)
    )
    new_vel = Point(
        v0.x + (dt/6)*(k1_v.x + 2*k2_v.x + 2*k3_v.x + k4_v.x),
        v0.y + (dt/6)*(k1_v.y + 2*k2_v.y + 2*k3_v.y + k4_v.y),
        v0.z + (dt/6)*(k1_v.z + 2*k2_v.z + 2*k3_v.z + k4_v.z)
    )
    body.location = new_loc
    body.velocity = new_vel

def simulate_RK4(earth, leader, follower, dt, steps, axis='y'): # !! Change axis to desired axis !!
    mu = 6.67408e-11 * earth.mass
    leader_data = {'pos': [], 'elems': []}
    follower_data = {'pos': [], 'elems': []}
    sep_axis = []

    for _ in range(steps):
        rk4_step(leader, earth, dt)
        rk4_step(follower, earth, dt)

        if axis == 'x':
            d = follower.location.x - leader.location.x
        elif axis == 'y':
            d = follower.location.y - leader.location.y
        else:
            d = follower.location.z - leader.location.z

        leader_data['pos'].append((leader.location.x, leader.location.y, leader.location.z))
        follower_data['pos'].append((follower.location.x, follower.location.y, follower.location.z))
        leader_data['elems'].append(orbital_elements(leader, mu))
        follower_data['elems'].append(orbital_elements(follower, mu))
        sep_axis.append(d)

    return leader_data, follower_data, sep_axis

def plot_orbital_elements(times, *elem_lists, labels=None):
    keys = ['a', 'e', 'i', 'RAAN', 'omega', 'nu']
    num = len(elem_lists)
    arrs = []
    for elems in elem_lists:
        arr = np.array([[e[k] for k in keys] for e in elems])
        arrs.append(arr)

    labels = labels or [f'Set {i+1}' for i in range(num)]
    elem_labels = ['a (m)', 'e', 'i (deg)', 'Ω (deg)', 'ω (deg)', 'ν (deg)']
    plt.figure(figsize=(14, 4*num))
    for idx in range(6):
        plt.subplot(6, 1, idx+1)
        for arr, lab in zip(arrs, labels):
            plt.plot(times, arr[:, idx], label=lab)
        plt.ylabel(elem_labels[idx])
        plt.grid(True)
        plt.legend()
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def run_simulation(earth, leader, follower, impulse_axis, impulse_sign, dt, steps, separation_axis='y'): # !! Change separation axis to desired axis !!
    
    leader_copy = copy.deepcopy(leader)
    follower_copy = copy.deepcopy(follower)

    # Edit impulse here
    k = 2960    # N/m, SERPENT = 2960
    x = 0.006   # Meters compression, SERPENT = 0.006
    m = follower_copy.mass
    delta_v = 0.5*math.sqrt((k * x ** 2) / m)

    if impulse_axis == 'x':
        follower_copy.velocity.x += impulse_sign * delta_v
    elif impulse_axis == 'y':
        follower_copy.velocity.y += impulse_sign * delta_v
    elif impulse_axis == 'z':
        follower_copy.velocity.z += impulse_sign * delta_v

    return simulate_RK4(earth, leader_copy, follower_copy, dt, steps, axis=separation_axis)

def main():
    earth = Body("Earth", 5.972e24, Point(0,0,0), Point(0,0,0))
    orbit_radius = 6828000
    G = 6.67408e-11
    v_circular = math.sqrt(G * earth.mass / orbit_radius)
    init_loc = Point(orbit_radius, 0, 0)
    init_vel = Point(0, v_circular, 0)

    leader = Body("Leader", 2.66, init_loc, init_vel)
    follower = Body("Follower", 1.33, init_loc, init_vel)

    # Simulation time parameters
    dt = 300              # 1 second is baseline, adjust to 60 for minute, up to dt = 300 for 5 min.
    steps = 864           # Scale with dt based on length of time desired.

    # Choose axis
    axis = 'y'

    ld, fd_pos, sep_pos = run_simulation(earth, leader, follower, impulse_axis=axis, impulse_sign=+1, dt=dt, steps=steps, separation_axis=axis)
    _, fd_neg, sep_neg = run_simulation(earth, leader, follower, impulse_axis=axis, impulse_sign=-1, dt=dt, steps=steps, separation_axis=axis)


    times = [i*dt for i in range(steps)]

    plt.figure(figsize=(10,6))
    plt.plot(times, sep_pos, label='+ separation')
    plt.plot(times, sep_neg, label='- separation')
    plt.xlabel('Time (s)')
    plt.ylabel('Signed separation (m)')
    plt.title('Impulse-Driven Signed Separation')
    plt.grid(True)
    plt.legend()
    plt.show()

    plot_orbital_elements(
        times,
        ld['elems'], fd_pos['elems'], fd_neg['elems'],
        labels=['Leader', 'Follower +z', 'Follower -z']
    )

if __name__ == '__main__':
    main()
