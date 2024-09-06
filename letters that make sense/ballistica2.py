#for all deviations and calc target
import math
import numpy as np
import matplotlib.pyplot as plt

def bullet_trajectory(initial_velocity, angle):
    g = 9.81
    angle_rad = math.radians(angle)
    max_height = (initial_velocity ** 2) * (math.sin(angle_rad) ** 2) / (2 * g)
    return max_height

def find_distance(initial_velocity, angle):
    g = 9.81
    angle_rad = math.radians(angle)
    range_distance = (initial_velocity ** 2) * math.sin(2 * angle_rad) / g
    return range_distance

def plot_trajectory(initial_velocity, angle):
    g = 9.81
    angle_rad = math.radians(angle)
    time_of_flight = (2 * initial_velocity * math.sin(angle_rad)) / g
    times = np.linspace(0, time_of_flight, num=100)
    heights = []

    for t in times:
        height = (initial_velocity * math.sin(angle_rad) * t) - (0.5 * g * t ** 2)
        heights.append(height)

    plt.figure(figsize=(10, 5))
    plt.plot(times, heights, label='Bullet Trajectory', color='blue')
    plt.axhline(0, color='red', linestyle='--', label='Ground Level')
    plt.title('Parabola of Bullet Trajectory')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.xlim(0, time_of_flight)
    plt.ylim(0, max(heights) * 1.1)
    plt.grid()
    plt.legend()
    plt.show()

initial_velocity = float(input("Enter the initial velocity of the bullet (m/s): "))
angle = float(input("Enter the angle of the shot (degrees): "))
weight = float(input("Enter the weight of the bullet (grams): "))
target_distance = float(input("Enter the target distance (meters): "))

weight_kg = weight / 1000
max_height = bullet_trajectory(initial_velocity, angle)
calculated_distance = find_distance(initial_velocity, angle)

print(f"Weight of the bullet: {weight_kg:.3f} kg")
print(f"Maximum deviation of the bullet from the ground: {max_height:.5f} meters")
print(f"Calculated target distance: {calculated_distance:.5f} meters")
print(f"Entered target distance: {target_distance:.5f} meters")

plot_trajectory(initial_velocity, angle)