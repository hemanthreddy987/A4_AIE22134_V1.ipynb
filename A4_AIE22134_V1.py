import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Load the dataset
# Assuming the dataset is stored in a CSV file named "traffic_data.csv"
data = pd.read_csv("futuristic_city_traffic.csv")

# Extracting necessary features
speed = data['Speed']
is_peak_hour = data['Is Peak Hour']

# Separating data based on peak hour indicator
speed_peak_hour = speed[is_peak_hour == 1]
speed_non_peak_hour = speed[is_peak_hour == 0]

# Calculate mean for each class (centroid)
mean_peak_hour = np.mean(speed_peak_hour)
mean_non_peak_hour = np.mean(speed_non_peak_hour)

# Calculate spread (standard deviation) for each class
spread_peak_hour = np.std(speed_peak_hour)
spread_non_peak_hour = np.std(speed_non_peak_hour)

# Calculate distance between mean vectors between classes
distance = np.linalg.norm(mean_peak_hour - mean_non_peak_hour)

print("Mean speed during peak hour:", mean_peak_hour)
print("Standard deviation of speed during peak hour:", spread_peak_hour)
print()
print("Mean speed during non-peak hour:", mean_non_peak_hour)
print("Standard deviation of speed during non-peak hour:", spread_non_peak_hour)
print()
print("Distance between mean vectors of classes:", distance)



#2


# Load the dataset
# Assuming the dataset is stored in a CSV file named "traffic_data.csv"

# Extract the 'Speed' feature
speed = data['Speed']

# Plot histogram
plt.hist(speed, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Speed')
plt.ylabel('Frequency')
plt.title('Histogram of Vehicle Speeds')
plt.grid(True)
plt.show()

# Calculate mean and variance
mean_speed = np.mean(speed)
variance_speed = np.var(speed)

print("Mean speed:", mean_speed)
print("Variance of speed:", variance_speed)


#3


# Load the dataset
# Assuming the dataset is stored in a CSV file named "traffic_data.csv"


# Randomly select two feature vectors
feature_vector1 = data.sample(n=1, random_state=1).iloc[0]
feature_vector2 = data.sample(n=1, random_state=2).iloc[0]

# Extract features from the selected vectors
features1 = feature_vector1[['Speed', 'Traffic Density']].values
features2 = feature_vector2[['Speed', 'Traffic Density']].values

# Calculate Minkowski distance for r from 1 to 10
r_values = range(1, 11)
distances = []
for r in r_values:
    distance = np.linalg.norm(features1 - features2, ord=r)
    distances.append(distance)

# Plot the distances
plt.plot(r_values, distances, marker='o', linestyle='-')
plt.xlabel('r')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance vs. r')
plt.grid(True)
plt.show()


#4


# Load the dataset
# Assuming the dataset is stored in a CSV file named "traffic_data.csv"


# Define features (X) and target variable (y)
X = data[['Speed', 'Is Peak Hour']]
y = data['Traffic Density']  # Assuming 'Traffic Density' is the target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shapes of the training and test sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

#5


# Create kNN classifier with k=3
# Assuming your data is stored in a DataFrame called 'data'
# Extracting features and target variable
X = data[['Speed', 'Is Peak Hour']]
y = data['Traffic Density']  # Replace 'Target_Variable' with the name of your target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the kNN classifier
neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(X_train, y_train)


#6
# Test the accuracy of the kNN classifier using the test set
# Test the accuracy of the kNN classifier
accuracy = neigh.score(X_test, y_test)
print("Accuracy:", accuracy)

#7
# Choose a test vector (for example, the first vector in the test set)
test_vector = X_test.iloc[0]  # Assuming X_test is a DataFrame

# Perform classification for the chosen test vector
predicted_class = neigh.predict([test_vector])

print("Predicted class label for the test vector:", predicted_class)


#8


# Lists to store accuracy values
accuracies_k3 = []
accuracies_k1 = []

# Vary k from 1 to 11
for k in range(1, 12):
    # Train kNN classifier with k=3
    neigh_k3 = KNeighborsRegressor(n_neighbors=3)
    neigh_k3.fit(X_train, y_train)
    accuracy_k3 = neigh_k3.score(X_test, y_test)
    accuracies_k3.append(accuracy_k3)

    # Train NN classifier with k=1
    neigh_k1 = KNeighborsRegressor(n_neighbors=1)
    neigh_k1.fit(X_train, y_train)
    accuracy_k1 = neigh_k1.score(X_test, y_test)
    accuracies_k1.append(accuracy_k1)

# Plot the accuracies
plt.plot(range(1, 12), accuracies_k3, label='kNN (k=3)')
plt.plot(range(1, 12), accuracies_k1, label='NN (k=1)')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. k for kNN and NN classifiers')
plt.legend()
plt.grid(True)
plt.show()
