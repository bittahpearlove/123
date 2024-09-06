#for bullet drop
import math

def calculate_drop(target_distance, initial_velocity, gravity=9.81):
    time_of_flight = target_distance / initial_velocity
    drop = 0.5 * gravity * (time_of_flight ** 2)
    return drop

def main():
    print("Ballistic Drop Calculator")
    initial_velocity = float(input("Enter the initial velocity (m/s): "))
    bullet_weight = float(input("Enter the bullet weight (grams): "))
    bullet_weight_kg = bullet_weight / 1000.0
    target_distances = [100, 200, 300, 400, 500]
    
    print("\nVertical Drop for Target Distances:")
    for distance in target_distances:
        drop = calculate_drop(distance, initial_velocity)
        print(f"Distance: {distance} meters -> Vertical Drop: {drop:.4f} meters (Bullet Weight: {bullet_weight} grams)")

if __name__ == "__main__":
    main()