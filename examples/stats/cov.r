# Examples of empirical correlation matrix computation using cov function

## Read chains

chains <- read.table(file="chain01.csv", header=FALSE, sep=",")

## Compute correlation matrix

print(cov(chains))
