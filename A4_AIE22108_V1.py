import os
import cv2
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
# Define the path to the directory containing the training data
train_data_path = r"train"

# Initialize lists to store file paths and labels
filepaths = []
labels = []

# Get the list of subdirectories (class labels)
folds = os.listdir(train_data_path)

# Iterate over each subdirectory
for fold in folds:
    # Get the full path to the subdirectory
    f_path = os.path.join(train_data_path, fold)
    # Get the list of file names in the subdirectory
    filelists = os.listdir(f_path)
    
    # Iterate over each file in the subdirectory
    for file in filelists:
        # Get the full path to the file
        filepaths.append(os.path.join(f_path, file))
        # Store the label (subdirectory name) for the file
        labels.append(fold)

# Initialize a list to store image vectors
images = []

# Iterate over each file path
for filepath in filepaths:
    # Read the image from the file
    img = cv2.imread(filepath)
    # Resize the image to a fixed size
    img = cv2.resize(img, (100, 100))  # Adjust the size as needed
    # Flatten the image into a 1D array
    img_vector = img.flatten()
    # Append the flattened image vector to the list
    images.append(img_vector)

# Convert the list of image vectors to a numpy array
images_array = np.array(images)

# Create a DataFrame to store the image vectors and labels
df = pd.DataFrame(images_array, columns=[f"pixel_{i}" for i in range(images_array.shape[1])])
df['label'] = labels

print("Shape of DataFrame:", df.shape)
print("Head of DataFrame:", df.head())

# Separate data into two classes: "normal" and "OSSC"
normal_class = df[df['label'] == 'Normal']
oscc_class = df[df['label'] == 'OSCC']

print("Shape of Normal class:", normal_class.shape)
print("Shape of OSCC class:", oscc_class.shape)
# Check unique values in the 'label' column
unique_labels = df['label'].unique()
print(unique_labels)
# Calculate the mean for each class
normal_mean = normal_class.iloc[:, :-1].mean(axis=0)
oscc_mean = oscc_class.iloc[:, :-1].mean(axis=0)

print("Mean for Normal class:", normal_mean)
print("Mean for OSSC class:", oscc_mean)
# Drop the "label" column before calculating the standard deviation
normal_class_no_label = normal_class.drop(columns=['label'])
ossc_class_no_label = oscc_class.drop(columns=['label'])

# Calculate standard deviation for each class
std_normal = np.std(normal_class_no_label, axis=0)  # Assuming normal_class is the DataFrame for the "Normal" class
std_ossc = np.std(ossc_class_no_label, axis=0)      # Assuming ossc_class is the DataFrame for the "OSCC" class

# Calculate mean vectors for each class
mean_normal = np.mean(normal_class_no_label, axis=0)
mean_ossc = np.mean(ossc_class_no_label, axis=0)

# Calculate distance between mean vectors
distance_between_means = np.linalg.norm(mean_normal - mean_ossc)

# Print results
print("Standard Deviation for Normal class:", std_normal)
print("Standard Deviation for OSSC class:", std_ossc)
print("Distance between mean vectors:", distance_between_means)


# Assuming you have loaded your dataset into a DataFrame called "data"
#A2
# Selecting a feature from the dataset
feature_name = "pixel_0"  # Example feature name

# Extracting the feature values
feature_values = df[feature_name]

# Calculating histogram
hist, bins = np.histogram(feature_values, bins=10)  # Adjust the number of bins as needed

# Plotting histogram
plt.hist(feature_values, bins=bins, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of ' + feature_name)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculating mean and variance
mean_value = np.mean(feature_values)
variance_value = np.var(feature_values)

print("Mean of", feature_name + ":", mean_value)
print("Variance of", feature_name + ":", variance_value)
#A3
# Assuming you have loaded your dataset into a DataFrame called "data"

# Selecting two feature vectors
feature_vector1 = df.iloc[0, :-1]  # Example: First row, excluding the label column
feature_vector2 = df.iloc[1, :-1]  # Example: Second row, excluding the label column

# Normalize feature vectors
normalized_feature_vector1 = (feature_vector1 - feature_vector1.mean()) / feature_vector1.std()
normalized_feature_vector2 = (feature_vector2 - feature_vector2.mean()) / feature_vector2.std()

# Calculate Minkowski distance with r from 1 to 10
r_values = range(1, 11)
distances = []

for r in r_values:
    distance_r = distance.minkowski(normalized_feature_vector1, normalized_feature_vector2, p=r)
    distances.append(distance_r)

# Plotting the distances
plt.plot(r_values, distances, marker='o', linestyle='-')
plt.title('Minkowski Distance vs. r')
plt.xlabel('r')
plt.ylabel('Minkowski Distance')
plt.grid(True)
plt.show()
#A4

# Assuming X contains your feature vectors and y contains the corresponding class labels
# Assuming 'df' is your DataFrame containing the dataset
X = df.drop('label', axis=1)  # Features (pixel_0 to pixel_29999)
y = df['label']  # Class labels
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the shape of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

#A5

# Assuming X_train and y_train are your training feature vectors and corresponding class labels
# Assuming X_test and y_test are your testing feature vectors and corresponding class labels

# Initialize the kNN classifier with k=3
neigh = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
neigh.fit(X_train, y_train)

# Predict the class labels for the test data
y_pred = neigh.predict(X_test)

# Evaluate the classifier
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

#A6
# Assuming 'neigh' is your trained kNN classifier
accuracy = neigh.score(X_test, y_test)
print("Accuracy:", accuracy)
#A7
# Assuming 'neigh' is your trained kNN classifier
accuracy = neigh.score(X_test, y_test)
print("Accuracy:", accuracy)
# Assuming 'neigh' is your trained kNN classifier
predictions = neigh.predict(X_test)
print("Predictions:", predictions)
#A8



# Define the range of k values
k_values = np.arange(1, 12)

# Initialize lists to store accuracies for NN and kNN classifiers
accuracy_nn = []
accuracy_knn = []

# Train and test the classifiers for each value of k
for k in k_values:
    # Train NN classifier
    nn_classifier = KNeighborsClassifier(n_neighbors=1)
    nn_classifier.fit(X_train, y_train)
    nn_pred = nn_classifier.predict(X_test)
    accuracy_nn.append(accuracy_score(y_test, nn_pred))
    print(f"Accuracy for Nearest Neighbor (k=1) with k={k}: {accuracy_nn[-1]}")
    
    # Train kNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    knn_pred = knn_classifier.predict(X_test)
    accuracy_knn.append(accuracy_score(y_test, knn_pred))
    print(f"Accuracy for kNN (k={k}) with k={k}: {accuracy_knn[-1]}")

# Plotting the accuracies
plt.plot(k_values, accuracy_nn, label='Nearest Neighbor (k=1)')
plt.plot(k_values, accuracy_knn, label='kNN (k=3)')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. k for Nearest Neighbor and kNN classifiers')
plt.legend()
plt.show()
#A9


# Predictions on training data
y_train_pred = knn_classifier.predict(X_train)
# Predictions on test data
y_test_pred = knn_classifier.predict(X_test)

# Confusion matrix for training data
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix for Training Data:")
print(conf_matrix_train)

# Confusion matrix for test data
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix for Test Data:")
print(conf_matrix_test)

# Classification report for training data
print("\nClassification Report for Training Data:")
print(classification_report(y_train, y_train_pred))

# Classification report for test data
print("\nClassification Report for Test Data:")
print(classification_report(y_test, y_test_pred))

