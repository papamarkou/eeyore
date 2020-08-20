# Compute INSE Monte Carlo covariance estimate using mcse.initseq function of mcmcse package

## Load libraries

library(mcmcse)

## Read chains

chains <- read.table(file="chain01.csv", header=FALSE, sep=",")

## Compute INSE Monte Carlo covariance estimate

print(mcse.initseq(chains)$cov)

## Compute adjusted INSE Monte Carlo covariance estimate

print(mcse.initseq(chains, adjust=TRUE)$cov.adj)
