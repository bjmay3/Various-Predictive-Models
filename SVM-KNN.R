# SVM exercise

# Load the "kernlab" library
library(kernlab)

# Load the data and matricize it
data = read.table("credit_card_data-headers.txt", header = TRUE, sep = "\t")
data = as.matrix(data)

# Find the classifier
model = ksvm(data[, 1:10], data[, 11], type = "C-svc", kernel = "vanilladot"
             , C = 100, scaled = TRUE)

# Calculate a1 to am
a = colSums(model@xmatrix[[1]] * model@coef[[1]])
a

# Calculate a0
a0 = -model@b
a0

# See what the model predicts
pred = predict(model, data[, 1:10])
pred

# View Confusion Matrix and see what percentage of the model's prediction
# matches the actual data
cm = table(pred, data[, 11])
cm
sum(pred == data[, 11]) / nrow(data)


# K-NN exercise

# Load the "kknn" library
library(kknn)

# Load the data
data = read.table("credit_card_data-headers.txt", header = TRUE, sep = "\t")

# This next sequence of steps iterates through k-values from 1 to 100 to
# determine which one provides the most accurate results.
m = nrow(data)  # Counts the number of rows in the dataset.
compare = 0  # Initializes variable for comparing results with different k-values
for (knn in 1:100) {  # Iterate k-values from 1 to 100
        count = 0  # Initialize variable to count when predicted = actual
        for (i in 1:m) {  # Iterate through all datapoints in dataset
                TrainData = data[-i,]  # Training set all but i'th datapoint
                TestData = data[i,]  # Test set is the i'th datapoint
                # Fit the results using "kknn"
                fit = kknn(R1 ~ ., TrainData, TestData, k=knn, scale=TRUE)
                # Extract predicted value for i'th datapoint
                pred = round(fitted.values(fit))  # Rounding necessary
                # Extract actual response for i'th datapoint from dataset
                act = TestData[, 11]
                if (pred == act) {  # If predicted = actual then increment count
                        count = count + 1
                }
        }
        if (compare < count / m) {  # count / m is prediction accuracy
                compare = count / m # "compare" will end up at highest accuracy
                best_k = knn # Capture k-value of highest accuracy prediction
        }
}
# Print out k-value and percentage value of highest accuracy prediction
best_k
compare

