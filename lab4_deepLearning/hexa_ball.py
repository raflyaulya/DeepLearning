import pygame
import math

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Bouncing Ball in Spinning Hexagon")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Hexagon parameters
center_x, center_y = width // 2, height // 2
hex_radius = 200
rotation_speed = 2  # radians per second
rotation_angle = 0.0

# Ball parameters
ball_radius = 10
ball_color = RED
ball_x, ball_y = center_x, center_y  # Start at center
ball_vx, ball_vy = 100.0, 0.0  # Initial velocity (pixels per second)

# Physics parameters
gravity = 400  # pixels per second squared
friction = 0.7  # per second (velocity multiplier)
restitution = 0.8  # Bounce coefficient

def closest_point_on_segment(A, B, P):
    ax, ay = A
    bx, by = B
    px, py = P

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    t = (apx * abx + apy * aby) / (abx**2 + aby**2 + 1e-6)
    t = max(0.0, min(1.0, t))

    closest_x = ax + abx * t
    closest_y = ay + aby * t
    return (closest_x, closest_y)

running = True
while running:
    dt = clock.tick(60) / 1000.0  # Delta time in seconds

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update hexagon rotation
    rotation_angle += rotation_speed * dt

    # Apply physics to the ball
    ball_vy += gravity * dt
    ball_vx *= (1 - friction * dt)
    ball_vy *= (1 - friction * dt)

    ball_x += ball_vx * dt
    ball_y += ball_vy * dt

    # Calculate hexagon vertices
    vertices = []
    for i in range(6):
        theta = rotation_angle + i * math.pi / 3
        x = center_x + hex_radius * math.cos(theta)
        y = center_y + hex_radius * math.sin(theta)
        vertices.append((x, y))

    # Check collisions with each wall
    for i in range(6):
        A = vertices[i]
        B = vertices[(i + 1) % 6]
        P = (ball_x, ball_y)

        closest = closest_point_on_segment(A, B, P)
        dx = ball_x - closest[0]
        dy = ball_y - closest[1]
        distance = math.hypot(dx, dy)

        if distance <= ball_radius:
            # Calculate normal vector (from midpoint to center)
            M = ((A[0] + B[0])/2, (A[1] + B[1])/2)
            normal_x = center_x - M[0]
            normal_y = center_y - M[1]
            length = math.hypot(normal_x, normal_y)
            if length == 0:
                continue
            normal_x /= length
            normal_y /= length

            # Wall velocity at collision point
            cp_x, cp_y = closest
            V_wall_x = -rotation_speed * (cp_y - center_y)
            V_wall_y = rotation_speed * (cp_x - center_x)

            # Relative velocity
            V_rel_x = ball_vx - V_wall_x
            V_rel_y = ball_vy - V_wall_y

            dot = V_rel_x * normal_x + V_rel_y * normal_y

            if dot < 0:  # Only collide if moving towards the wall
                # Apply impulse
                j = -(1 + restitution) * dot
                ball_vx += j * normal_x
                ball_vy += j * normal_y

                # Correct position to prevent overlap
                penetration = ball_radius - distance
                ball_x += normal_x * penetration
                ball_y += normal_y * penetration

    # Draw everything
    screen.fill(BLACK)
    pygame.draw.lines(screen, WHITE, True, vertices, 2)
    pygame.draw.circle(screen, ball_color, (int(ball_x), int(ball_y)), ball_radius)
    pygame.display.flip()

pygame.quit()