# Compute multivariate ESS using multiESS function of mcmcse package

## Load libraries

library(mcmcse)

## Read chains

chains <- read.table(file="chain01.csv", header=FALSE, sep=",")

## Compute multivariate ESS using INSE MC covariance estimation

print(multiESS(chains, covmat=mcse.initseq(chains)$cov))

## Compute multivariate ESS batch mean MC covariance estimation

print(multiESS(chains))

# print(multiESS(chains, covmat=mcse.multi(chains)$cov)) # Same as print(multiESS(chains))
