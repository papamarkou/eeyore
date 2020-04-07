p <- function(j, i, n, b, normalised=TRUE) {
  eb <- exp(-b)
  numerator <- eb^abs(j-i)
  
  result <- ifelse(normalised, numerator/(eb*(-eb^i-eb^(n-i)+2)/(1-eb)), numerator)

  return(result)
}

pa <- function(i, b, n) {
  eb <- exp(-b)
  return(eb*(-eb^i-eb^(n-i)+2)/(1-eb))
}

normaliser <- pa(4, 0.5, 10)

print(paste("Normalising constant =", normaliser, sep=' '))

s_not_normalised <- 0
for (j in c(0, 1, 2, 3, 5, 6, 7, 8, 9, 10)) {
  s_not_normalised <- s_not_normalised + p(j, 4, 10, 0.5, normalised=FALSE)
}

s_normalised <- 0
for (j in c(0, 1, 2, 3, 5, 6, 7, 8, 9, 10)) {
  s_normalised <- s_normalised + p(j, 4, 10, 0.5)
}

print(paste("Manual sum of all probabilities =", s_not_normalised, sep=' '))

print(paste("Manual sum divided by normalising constant is one:", s_not_normalised/normaliser, sep=' '))

print(paste("Sum of normalised probabilities =", s_normalised, sep=' '))
