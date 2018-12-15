# Cross-Validation exercise (K-NN)

# Load the "kknn" library
library(kknn)

# Load the data
data = read.table("credit_card_data-headers.txt", header = TRUE, sep = "\t")

# Choose value for kmax
kmax = 100

# Develop classifier algorithm using "train.kknn", display results, and plot
classifier = train.kknn(R1 ~ ., data = data, kmax = kmax, kernel = c('triangular', 
                        'rectangular', 'optimal'), scale = TRUE)
classifier
plot(classifier)

# Calculate predicted values, round to 0 or 1, and display them
pred = as.integer(fitted(classifier)[[61]][1:nrow(data)] + .5)
pred

# Calculate the accuracy for this particular classifier
accuracy = sum(pred == data$R1) / nrow(data)
accuracy

# Calculate accuracy over full range of k-values and find best accuracy
accuracy = rep(0, kmax) # Initializes the accuracy vector with all zeroes
for (k in 1:kmax) { # Loop over all k-values
        # Calculate predicted values and round to 0 or 1
        pred = as.integer(fitted(classifier)[[k]][1:nrow(data)] + .5)
        # Calculate accuracy for each k-value
        accuracy[k] = sum(pred == data$R1) / nrow(data)
}
# Display accuracy over all k-values, determine maximum accuracy, and find that
# k-value. Plot accuracy vs. k-values
accuracy
max(accuracy)
which.max(accuracy)
plot(accuracy, xlab = "K-Value")


# Splitting data into training, validation, & test sets (K-NN & SVM)

# Load the "kknn" and "kernlab" libraries
library(kknn)
library(kernlab)

# Load the data
data = read.table("credit_card_data-headers.txt", header = TRUE, sep = "\t")

# Set seed, divide data into training, validation, and test sets (60%/20%/20%)
set.seed(1)
m = nrow(data) # Counts the number of rows in the dataset
# Sets up a variable to randomly select 40% of the data
split = sample(1:m, size = round(m * .4), replace = FALSE)
train_data = data[-split, ] # 60% of data in training set
testval_data = data[split, ] # 40% of data in test & validation sets
mtestval = nrow(testval_data) # Counts number of rows in test & val set
# Sets up a variable to randomly select 50% of the test & val data
split_tv = sample(1:mtestval, size = round(mtestval / 2), replace = FALSE)
val_data = testval_data[-split_tv, ] # 50% of test & val data to validation set
test_data = testval_data[split_tv, ] # 50% of test & val data to test set

# Build the K-NN model

# Choose value for kmax
kmax = 100

# Develop classifier algorithm using "train.kknn", display results, and plot
classifier_knn = train.kknn(R1 ~ ., data = train_data, kmax = kmax, kernel = c(
        'triangular', 'rectangular', 'optimal'), scale = TRUE)
classifier_knn
plot(classifier_knn)
# kernel = optimal; k = 38

# Build the SVM model

# Develop classifier algorithm using "ksvm" and display
classifier_svm = ksvm(R1 ~ ., train_data, type = "C-svc", kernel = "vanilladot"
                  , C = 100, scaled = TRUE)
classifier_svm
# kernel = vanilladot; C = 100

# Compare best K-NN model to best SVM model using validation set

# Make predictions with knn classifier using the validation data, display
pred_knn = round(predict(classifier_knn, val_data)) # Rounding required
pred_knn

# Generate confusion matrix and calculate accuracy
cm_knn = table(pred_knn, val_data$R1)
cm_knn
accuracy_knn = sum(pred_knn == val_data$R1) / nrow(val_data)
accuracy_knn
# accuracy = 0.8779

# Make predictions with svm classifier using the validation dataset, display
pred_svm = predict(classifier_svm, val_data[, 1:10])
pred_svm

# Generate confusion matrix and calculate accuracy
cm_svm = table(pred_svm, val_data[, 11])
cm_svm
accuracy_svm = sum(pred_svm == val_data[, 11]) / nrow(val_data)
accuracy_svm
# accuracy = 0.8702
# K-NN model has better accuracy so choose it

# Estimate performance of K-NN model with test set

# Make predictions with knn classifier using the test data, display
pred_knn = round(predict(classifier_knn, test_data)) # Rounding required
pred_knn

# Generate confusion matrix and calculate accuracy
cm_knn = table(pred_knn, test_data$R1)
cm_knn
accuracy_knn = sum(pred_knn == test_data$R1) / nrow(test_data)
accuracy_knn
# accuracy = 0.8626


# K-means clustering problem

# Load the data
data = read.table("iris.txt", header = TRUE, sep = "", row.names = 1)

# Use the elbow method to find the optimal # of clusters
set.seed(1)
# Initialize vector to capture within cluster sum of squares (WCSS).
# This metric will determine optimal number of clusters
wcss = vector()
# Loop through possible k-values 1 to 10. For each k, capture the total WCSS
for (i in 1:10) {
        wcss[i] = sum(kmeans(data[, 1:4], centers = i)$withinss)
}
# Plot the total WCSS against the k-value to find elbow
plot(1:10, wcss, type = "b", main = "Optimal Cluster Determination", xlab = 
             "Number of Clusters", ylab = "Total WCSS")
# Elbow occurs at k = 3

# Apply the K-Means clustering function using k = 3
set.seed(1)
model = kmeans(data[, 1:4], centers = 3)

# Visualizing the K-Means Clustering
library(cluster)
# Sepal length & width
clusplot(data[, 3:4], model$cluster, lines = 0, 
         shade = TRUE, color = TRUE, labels = 2, 
         plotchar = FALSE, span = TRUE, main = "Clusters", 
         xlab = "Petal Width", ylab = "Petal Length")
# Petal length & width
clusplot(data[, 1:2], model$cluster, lines = 0, 
         shade = TRUE, color = TRUE, labels = 2, 
         plotchar = FALSE, span = TRUE, main = "Clusters", 
         xlab = "Sepal Width", ylab = "Sepal Length")

# Analyze the predicted results vs. the actual results

# Load "plyr" library. Used to change predicted values
# from (1, 2, 3) to ("setosa", "virginica", "versicolor")
library(plyr)

# Create prediction vector that contains the species
# names as in the original data table
pred = model$cluster
pred = mapvalues(pred, from=c(1, 2, 3), to=c("setosa", "virginica", "versicolor"))

# Generate a confusion matrix
cm = matrix(nrow = nrow(data), ncol = 3) # Initializes confusion matrix
for (i in 1:nrow(data)) { # Runs through all rows & adds data appropriately
        cm[i, 1] = as.character(data$Species[i]) # Actual results
        cm[i, 2] = pred[i] # Predicted results
        # 1 if actual = predicted and 0 otherwise
        cm[i, 3] = sum(pred[i] == data$Species[i])
}
table(cm[, 1], cm[, 2]) # Print confusion matrix table
