#for angle
import math

def calculate_launch_angle(target_distance, initial_velocity, gravity=9.81):
    term = (gravity * target_distance) / (initial_velocity ** 2)
    if abs(term) > 1:
        return None  
    angle_rad = 0.5 * math.asin(term)
    angle_deg = math.degrees(angle_rad)
    return angle_deg
def main():
    print("Ballistic Calculator")
    initial_velocity = float(input("Enter the initial velocity (m/s): "))
    bullet_weight = float(input("Enter the bullet weight (grams): "))
    bullet_weight_kg = bullet_weight / 1000.0
    target_distances = [100, 200, 300, 400, 500]
    print("\nLaunch Angles for Target Distances:")
    for distance in target_distances:
        launch_angle = calculate_launch_angle(distance, initial_velocity)
        if launch_angle is not None:
            print(f"Distance: {distance} meters -> Launch Angle: {launch_angle:.2f} degrees (Bullet Weight: {bullet_weight} grams)")
        else:
            print(f"Distance: {distance} meters -> No valid launch angle exists (Bullet Weight: {bullet_weight} grams).")
if __name__ == "__main__":
    main()