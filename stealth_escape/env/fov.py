import numpy as np


def angle_between(pos, target):
    """Returns angle in degrees from pos looking at target."""
    dx = target[1] - pos[1]
    dy = target[0] - pos[0]
    return np.degrees(np.arctan2(dy, dx))


def normalize_angle(angle):
    """Normalize angle to [-180, 180]."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def is_in_fov(guard_pos, guard_facing, target_pos, fov_angle, fov_range):
    """
    Check if target_pos falls within guard's field of view cone.

    Args:
        guard_pos    : (row, col) of guard
        guard_facing : angle in degrees guard is facing
        target_pos   : (row, col) of target (agent)
        fov_angle    : total cone angle in degrees (e.g. 90)
        fov_range    : max distance in cells

    Returns:
        bool
    """
    dr = target_pos[0] - guard_pos[0]
    dc = target_pos[1] - guard_pos[1]
    dist = np.sqrt(dr ** 2 + dc ** 2)

    if dist > fov_range:
        return False

    angle_to_target = np.degrees(np.arctan2(dr, dc))
    diff = normalize_angle(angle_to_target - guard_facing)

    return abs(diff) <= fov_angle / 2


def has_line_of_sight(grid, pos1, pos2):
    """
    Bresenham line-of-sight check between two grid positions.
    Returns True if no walls block the path.

    Args:
        grid : 2D numpy array (1 = wall)
        pos1 : (row, col)
        pos2 : (row, col)
    """
    r1, c1 = pos1
    r2, c2 = pos2

    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    r, c = r1, c1
    n = 1 + dr + dc
    r_inc = 1 if r2 > r1 else -1
    c_inc = 1 if c2 > c1 else -1
    error = dr - dc
    dr *= 2
    dc *= 2

    for _ in range(n):
        if (r, c) != pos1 and (r, c) != pos2:
            if grid[r][c] == 1:   # wall
                return False
        if error > 0:
            r += r_inc
            error -= dc
        else:
            c += c_inc
            error += dr

    return True


def cast_rays(grid, pos, facing, fov_angle, fov_range, num_rays=8):
    """
    Cast rays from pos within FOV cone.
    Returns array of normalized distances to first wall/boundary hit.
    """
    half = fov_angle / 2
    angles = np.linspace(facing - half, facing + half, num_rays)
    distances = []

    rows, cols = grid.shape

    for angle in angles:
        rad = np.radians(angle)
        for d in np.linspace(0, fov_range, fov_range * 4):
            r = int(round(pos[0] + d * np.sin(rad)))
            c = int(round(pos[1] + d * np.cos(rad)))
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == 1:
                distances.append(d / fov_range)
                break
        else:
            distances.append(1.0)

    return np.array(distances, dtype=np.float32)