library(coda)

## Vectors

# Define two vectors

x <- c(1.3, 2.6, 9.2, 4.7)
y <- c(5.5, 1.1, 8.2, 0.4)

# Compute variance manually

sum((x-mean(x))^2)/(length(x)-1)

# Compute covariance bia built-in function

var(x)

# Compute covariance manually

(x-mean(x))%*%(y-mean(y))/(length(x)-1)

# Compute covariance via built-in function

cbind(c(var(x), cov(x, y)), c(cov(y, x), var(y)))

cov(cbind(x, y))

# Compute correlation manually

((x-mean(x))%*%(y-mean(y))/(length(x)-1))/sqrt((sum((x-mean(x))^2)/(length(x)-1))*(sum((y-mean(y))^2)/(length(y)-1)))

# Compute correlation via built-in function

cor(x, y)

# Compute correlation via coda

crosscorr(mcmc(cbind(x, y)))

## Emulate situation with two chains

# Define two more vectors

z <- c(9.9, 4.6, 1.3, 2.1)
w <- c(2.5, 0.9, 0.2, 1.4)

# Compute correlation with built-in function

mean(c(cor(x, y), cor(z, w)))

cor(c(x, z), c(y, w))

crosscorr(mcmc.list(list(mcmc(cbind(x=x, y=y)), mcmc(cbind(x=z, y=w)))))
