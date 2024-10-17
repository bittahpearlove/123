import pandas as pd
import numpy as np
import os

# Function to generate synthetic data
def generate_synthetic_data(num_samples):
    # Define possible values for categorical variables
    network_types = ['2G', '3G', '4G']
    districts = ['Центр', 'Западный', 'Восточный', 'Северный', 'Южный', 'Губернский парк', 'Студенческий городок', 'Промышленная зона', 'Железнодорожный вокзал', 'Торговый центр']
    
    # Initialize lists to hold the generated data
    data = {
        'network_type': [],
        'district': [],
        'longitude': [],
        'latitude': [],
        'population_density': [],
        'user': [],
        'expected_quality': [],
        'actual_quality': []
    }
    
    for _ in range(num_samples):
        # Randomly select values for each column
        network_type = np.random.choice(network_types)
        district = np.random.choice(districts)
        longitude = np.random.uniform(54.45, 54.6)  # Random longitude between 54.5 and 54.6
        latitude = np.random.uniform(36.1, 36.45)    # Random latitude between 36.2 and 36.4
        population_density = np.random.randint(500, 1500)  # Random population density between 500 and 1500
        user = np.random.randint(50, 300)  # Random user count between 50 and 300
        expected_quality = np.random.randint(-100, -80)  # Random expected quality between -100 and -80
        actual_quality = expected_quality + np.random.randint(-10, 10)  # Actual quality close to expected
        
        # Append the generated values to the lists
        data['network_type'].append(network_type)
        data['district'].append(district)
        data['longitude'].append(longitude)
        data['latitude'].append(latitude)
        data['population_density'].append(population_density)
        data['user'].append(user)
        data['expected_quality'].append(expected_quality)
        data['actual_quality'].append(actual_quality)
    
    # Create a DataFrame from the generated data
    synthetic_df = pd.DataFrame(data)
    return synthetic_df

# Specify the path to the existing CSV file
csv_file_path = 'mobile_data_kaluga.csv'

# Check if the file exists
if os.path.exists(csv_file_path):
    # Load existing data
    existing_data = pd.read_csv(csv_file_path)
else:
    # If the file does not exist, create an empty DataFrame with the same columns
    existing_data = pd.DataFrame(columns=['network_type', 'district', 'longitude', 'latitude', 'population_density', 'user', 'expected_quality', 'actual_quality'])

# Generate new synthetic data
num_samples = 100  # Specify the number of samples you want to generate
new_data = generate_synthetic_data(num_samples)

# Concatenate existing data with new data
combined_data = pd.concat([existing_data, new_data], ignore_index=True)

# Save the combined data back to the CSV file
combined_data.to_csv(csv_file_path, index=False)

# Display the first few rows of the combined DataFrame
print(combined_data.head())
