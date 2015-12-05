import numpy as np
from numpy import linalg
from quadtree import QuadNode
# import scipy.integrate

N = 3000
R = 0.01

# *** 
# In principle we should have a closed form solution for the trajectory
# up to a collision (which we can predict)
# So we should have a perfectly solvable system with no interpenetration

def gen_x0(N):
    print 'Creating initial conditions'
    # Normal uniform speeds
    x0 = 1.0 * (np.random.rand(N, 4) - 0.5)
    # Grid positions
    sqrtNceil = int(np.ceil(np.sqrt(N)))
    for i in range(sqrtNceil):
        for j in range(sqrtNceil):
            if i * sqrtNceil + j > N - 1: break
            x0[i * sqrtNceil + j, 0:2] = [1.8 / sqrtNceil * i - 0.8, 1.8 / sqrtNceil * j - 0.8]
    return x0

t0 = 0
tend = 100
DELTA_T = 0.001
EPS_t = 1e-7
EPS_dt = 1e-3*0
EPS = 1e-6#1e-12
# HERE: INTRODUCE HERE A SPATIAL TOLERANCE for checkin collisions are out of bound errors
# TODO: Introduce a check after integration to check that the system is consistent
# TODO: Solve analytically particle trajectory (not a linearization)
# TODO: Would a potential-based solution be better? Instead of modelling collisions, have very high potentials?
#   This requires having very small steps however..

TEMPERATURE_COLD = 1.0
TEMPERATURE_HOT = 15.0
GRAVITY = -10.0
if GRAVITY == 0.0: print 'GRAVITY IS OFF.'

T = np.arange(t0, tend, DELTA_T)

print 'Running %s steps from t=%s to t=%s' % (len(T), t0, tend)

def solve_second_degree_polynomial(a, b, c):
    '''
    a x^2 + b x + c = 0
    '''
    if a == 0:
        if b == 0: return []
        return [-c / b]
    det = np.power(b, 2.0) - 4.0 * a * c
    if det >= 0.0:
        return [
            (-b - np.sqrt(det)) / (2.0 * a),
            (-b + np.sqrt(det)) / (2.0 * a)
        ]
    return []

def collisions(X, t, only_walls=False):
    quad = QuadNode((-1.0, -1.0, 1.0, 1.0))
    for i in range(N): quad.insert(i, X[i, 0], X[i, 1])

    evt = []
    evt_t = []
    for i in range(N):
        NEARBY_SIZE = 0.05

        # Add collision will all walls
        if X[i, 0] + EPS < -1 + R or X[i, 0] - EPS > 1 - R: print 'Particle %s out of bound! x=%s' % (i, X[i, 0]); #error
        if X[i, 1] + EPS < -1 + R or X[i, 1] - EPS > 1 - R: print 'Particle %s out of bound! y=%s' % (i, X[i, 1]); #error
        # Solve for dt
        # x + v * dt + 0.5 * GRAVITY * dt * dt == 1 - R or -1 + R
        # This can probably be done in a vector operation instead of a for loop
        ax = 0.5 * 0.0
        ay = 0.5 * GRAVITY
        bx = X[i, 2]
        by = X[i, 3]
        cxr = X[i, 0] - 1 + R
        cxl = X[i, 0] + 1 - R
        cyu = X[i, 1] - 1 + R
        cyd = X[i, 1] + 1 - R
        dt = []
        if cxr + 0.5 * NEARBY_SIZE > 0:
            dt += solve_second_degree_polynomial(ax, bx, cxr)
        if cxl - 0.5 * NEARBY_SIZE < 0:
            dt += solve_second_degree_polynomial(ax, bx, cxl)
        if cyu + 0.5 * NEARBY_SIZE > 0:
            dt += solve_second_degree_polynomial(ay, by, cyu)
        if cyu - 0.5 * NEARBY_SIZE < 0:
            dt += solve_second_degree_polynomial(ay, by, cyd)
        dt = np.array(dt)
        dt = dt[dt > EPS_dt]
        if len(dt) > 0:
            t_cols = t + dt
            evt.extend([(i, None)]*len(t_cols))
            evt_t.extend(t_cols)
        if only_walls: continue # Only walls

        nearby_box = (
            X[i, 0] - 0.5 * NEARBY_SIZE, 
            X[i, 1] - 0.5 * NEARBY_SIZE, 
            X[i, 0] + 0.5 * NEARBY_SIZE, 
            X[i, 1] + 0.5 * NEARBY_SIZE)
        nearby = quad.get_within(nearby_box)
        for j in range(N):
            if not i < j: continue
            if not j in nearby: continue

            dx = X[i, 0:2] - X[j, 0:2]
            dv = X[i, 2:4] - X[j, 2:4]

            # We should only test particles that are already more than 2R appart
            if linalg.norm(dx) < 2.0 * R - EPS and not only_walls: 
                print 'WARNING: (%s,%s) are within each other' % (i, j)
                print linalg.norm(dx)
                #error
                continue

            # We should only test for a crossing
            # in one direction. This could be a good optimisation?

            a = np.dot(dv, dv)
            if a == 0: continue
            b = 2.0 * np.dot(dv, dx)
            c = np.dot(dx, dx) - np.power(2.0 * R, 2.0)
            dt = np.array(solve_second_degree_polynomial(a, b, c))
            if dt is not None:
                # We need an EPS here, because if a collision is going to happen
                # in a very small dt, it's likely we just did handle the collision
                # at the step before. This could cause numerical resonance?
                dt = dt[dt > EPS_dt]
                if len(dt) > 0:
                    dt = np.min(dt)
                    t_col = t + dt
                    evt.append((i,j))
                    evt_t.append(t_col)

                    # if i==9 and j==8:
                    # print X[i, 0:2], X[i, 2:4]
                    # print X[j, 0:2], X[j, 2:4]
                    # print a * np.power(dt, 2.0) + b * dt + c
                    # print dt, X[i, 0:2] + X[i, 2:4] * dt, X[j, 0:2] + X[j, 2:4] * dt
                    # print (X[i, 0:2] + X[i, 2:4] * dt) - (X[j, 0:2] + X[j, 2:4] * dt)
                    # print np.sqrt(np.sum(np.power((X[i, 0:2] + X[i, 2:4] * dt) - (X[j, 0:2] + X[j, 2:4] * dt), 2.0))), dt, i, j
                    # print X[i, :]
                    # print X[j, :]
                    #print distance(i, j, X[:, 0:2] + X[:, 2:4] * dt)

    evt = np.array(evt)
    evt_t = np.array(evt_t)

    return evt[evt_t > t + EPS_t], evt_t[evt_t > t + EPS_t]

def periodic_boundary_conditions(X):
    X[X[:, 0] > 1, 0] = -1.0
    X[X[:, 0] < -1, 0] = 1.0
    X[X[:, 1] > 1, 1] = -1.0
    X[X[:, 1] < -1, 1] = 1.0
    return X

def reflective_boundary_conditions(X):

    ix = X[:, 0] > 1 - R
    X[ix, 2] *= -1.0
    #X[ix, 0] = 1.0 - R

    ix = X[:, 0] < -1 + R
    X[ix, 2] *= -1.0
    #X[ix, 0] = -1.0 + R

    ix = X[:, 1] > 1 - R
    X[ix, 3] -= TEMPERATURE_DIFF
    X[ix, 3] *= -1.0
    #X[ix, 1] = 1.0 - R

    ix = X[:, 1] < -1 + R
    X[ix, 3] *= -1.0
    X[ix, 3] += TEMPERATURE_DIFF
    #X[ix, 1] = -1.0 + R

    return X

def velocity_vervet(x_m, dt):
    x_p = x_m.copy()
    # velocity_vervet
    x_p[:, 0] = x_m[:, 0] + x_m[:, 2] * dt
    x_p[:, 1] = x_m[:, 1] + x_m[:, 3] * dt + 0.5 * GRAVITY * dt * dt
    x_p[:, 2] = x_m[:, 2]
    x_p[:, 3] = x_m[:, 3] + GRAVITY * dt
    return x_p

def distance(i, j, X):
    dx = X[i, 0:2] - X[j, 0:2]
    return linalg.norm(dx)

def simulate(x0, T):
    def done(r):
        r = np.array(r)
        np.save('last.npy', r)
        return r

    r = [x0] # TODO: Pre-allocate

    x_m = x0
    t_m = t0
    # Collision
    try:
        for stepIndex, t in enumerate(T):
            if stepIndex == 0: continue
            while True:
                # Predict collisions
                evt, evt_t = collisions(x_m, t_m, only_walls=False)
                if len(evt) > 0:
                    i_col = np.argmin(evt_t)
                    t_col = evt_t[i_col]
                    if t_col <= t:
                        dt = t_col - t_m
                        print 'Integrate (col) from t=%s to t_col=%s (dt=%s)' % (t_m, t_col, dt)
                        x_p = velocity_vervet(x_m, dt)
                        #r.append(x_p) # Optional
                        # Process all events
                        x_m = x_p.copy()
                        # TODO: Handle properly multiple events
                        for ij_col in evt[evt_t < t_col + EPS_t]:
                            if ij_col[1] is not None:
                                # Elastic collision for equal masses: switch speeds
                                # u1 = x_p[ij_col[0], 2:4]
                                # u2 = x_p[ij_col[1], 2:4]
                                # x_m[ij_col[0], 2:4] = u2
                                # x_m[ij_col[1], 2:4] = u1

                                # Normal unit vector
                                n_12 = x_p[ij_col[1], 0:2] - x_p[ij_col[0], 0:2]
                                n_12 = n_12 / np.linalg.norm(n_12)
                                # Relative velocities
                                rel_vel_12 = x_p[ij_col[1], 2:4] - x_p[ij_col[0], 2:4]
                                # Set
                                coef_restitution = 0.98
                                x_m[ij_col[0], 2:4] += 0.5 * (1 + coef_restitution) * np.dot(n_12, rel_vel_12) * n_12
                                x_m[ij_col[1], 2:4] -= 0.5 * (1 + coef_restitution) * np.dot(n_12, rel_vel_12) * n_12
                            else:
                                # Wall collision
                                i = ij_col[0]
                                pos = x_m[i, 0:2]
                                # Which boundary is closest
                                obj = np.array([
                                    abs(pos[0] - ( 1 - R)),  # right
                                    abs(pos[0] - (-1 + R)), # left
                                    abs(pos[1] - ( 1 - R)),  # top
                                    abs(pos[1] - (-1 + R))  # bottom
                                ])
                                ix = np.where(obj < EPS)[0]
                                #print i, ix, x_m[i, :], 'y=%s' % x_m[i, 1], 'vy=%s' % x_m[i, 3]
                                # Loop here because multiple boundaries might be hit
                                for ii in ix:
                                    if ii == 0 or ii == 1: x_m[i, 2] *= -1 # right/left
                                    # We use *uniform* distributions here
                                    if ii == 2: # top
                                        v = 2.0 * np.random.rand(2) - 1.0 # Random vector in [-1, 1]
                                        if v[1] > 0: v[1] = -v[1] # Vy must be negative
                                        x_m[i, 2:4] = v / linalg.norm(v) * TEMPERATURE_COLD # top
                                    if ii == 3: 
                                        v = 2.0 * np.random.rand(2) - 1.0 # Random vector in [-1, 1]
                                        if v[1] < 0: v[1] = -v[1] # Vy must be positive
                                        x_m[i, 2:4] = v / linalg.norm(v) * TEMPERATURE_HOT # top
                                    #print 'after', x_m[i, :], 'y=%s' % x_m[i, 1], 'vy=%s' % x_m[i, 3]
                        t_m = t_col
                    else: break
                else: break
            if t_m < t:
                print 'Integrate from t=%s to t=%s' % (t_m, t)
                dt = t - t_m
                x_p = velocity_vervet(x_m, dt)
                r.append(x_p)
                x_m = x_p
                t_m = t
            else: r.append(x_p)
    except KeyboardInterrupt:
        return done(r)

    return done(r)

def resume(r, T):
    if type(r) == str:
        r = np.load(r)
        print '%s rows loaded' % r.shape[0]
    x0 = r[-1, :, :]
    r2 = simulate(x0, T)[1:, :, :]
    r = np.concatenate((r, r2), axis=0)
    np.save('lastresume.npy', r)
    return r


from plot import plot_simulation

r = simulate(gen_x0(N), T)
plot_simulation(r[0::10, :, :])
