context('Test prediction of feature interactions')

set.seed(123)
n_threads <- 2

test_that("predict feature interactions works", {
  # simulate some binary data and a linear outcome with an interaction term
  N <- 1000
  P <- 5
  X <- matrix(rbinom(N * P, 1, 0.5), ncol = P, dimnames = list(NULL, letters[1:P]))
  # center the data (as contributions are computed WRT feature means)
  X <- scale(X, scale = FALSE)

  # outcome without any interactions, without any noise:
  f <- function(x) 2 * x[, 1] - 3 * x[, 2]
  # outcome with interactions, without noise:
  f_int <- function(x) f(x) + 2 * x[, 2] * x[, 3]
  # outcome with interactions, with noise:
  #f_int_noise <- function(x) f_int(x) + rnorm(N, 0, 0.3)

  y <- f_int(X)

  dm <- xgb.DMatrix(X, label = y, nthread = n_threads)
  param <- list(
    eta = 0.1, max_depth = 4, base_score = mean(y), lambda = 0, nthread = n_threads
  )
  b <- xgb.train(param, dm, 100)

  pred <- predict(b, dm, outputmargin = TRUE)

  # SHAP contributions:
  cont <- predict(b, dm, predcontrib = TRUE)
  expect_equal(dim(cont), c(N, P + 1))
  # make sure for each row they add up to marginal predictions
  expect_lt(max(abs(rowSums(cont) - pred)), 0.001)
  # Hand-construct the 'ground truth' feature contributions:
  gt_cont <- cbind(
      2. * X[, 1],
     -3. * X[, 2] + 1. * X[, 2] * X[, 3], # attribute a HALF of the interaction term to feature #2
      1. * X[, 2] * X[, 3]               # and another HALF of the interaction term to feature #3
     )
  gt_cont <- cbind(gt_cont, matrix(0, nrow = N, ncol = P + 1 - 3))
  # These should be relatively close:
  expect_lt(max(abs(cont - gt_cont)), 0.05)


  # SHAP interaction contributions:
  intr <- predict(b, dm, predinteraction = TRUE)
  expect_equal(dim(intr), c(N, P + 1, P + 1))
  # check assigned colnames
  cn <- c(letters[1:P], "(Intercept)")
  expect_equal(dimnames(intr), list(NULL, cn, cn))

  # check the symmetry
  expect_lt(max(abs(aperm(intr, c(1, 3, 2)) - intr)), 0.00001)

  # sums WRT columns must be close to feature contributions
  expect_lt(max(abs(apply(intr, c(1, 2), sum) - cont)), 0.00001)

  # diagonal terms for features 3,4,5 must be close to zero
  expect_lt(Reduce(max, sapply(3:P, function(i) max(abs(intr[, i, i])))), 0.05)

  # Intercept must have no interactions
  expect_lt(max(abs(intr[, 1:P, P + 1])), 0.00001)

  # interactions other than 2 x 3 must be close to zero
  intr23 <- intr
  intr23[, 2, 3] <- 0
  expect_lt(
    Reduce(max, sapply(1:P, function(i) max(abs(intr23[, i, (i + 1):(P + 1)])))),
    0.05
  )

  # Construct the 'ground truth' contributions of interactions directly from the linear terms:
  gt_intr <- array(0, c(N, P + 1, P + 1))
  gt_intr[, 2, 3] <- 1. * X[, 2] * X[, 3] # attribute a HALF of the interaction term to each symmetric element
  gt_intr[, 3, 2] <- gt_intr[, 2, 3]
  # merge-in the diagonal based on 'ground truth' feature contributions
  intr_diag <- gt_cont - apply(gt_intr, c(1, 2), sum)
  for (j in seq_len(P)) {
    gt_intr[, j, j] <- intr_diag[, j]
  }
  # These should be relatively close:
  expect_lt(max(abs(intr - gt_intr)), 0.1)
})

test_that("SHAP contribution values are not NAN", {
  d <- data.frame(
    x1 = c(-2.3, 1.4, 5.9, 2, 2.5, 0.3, -3.6, -0.2, 0.5, -2.8, -4.6, 3.3, -1.2,
           -1.1, -2.3, 0.4, -1.5, -0.2, -1, 3.7),
    x2 = c(291.179171, 269.198331, 289.942097, 283.191669, 269.673332,
           294.158346, 287.255835, 291.530838, 285.899586, 269.290833,
           268.649586, 291.530841, 280.074593, 269.484168, 293.94042,
           294.327506, 296.20709, 295.441669, 283.16792, 270.227085),
    y = c(9, 15, 5.7, 9.2, 22.4, 5, 9, 3.2, 7.2, 13.1, 7.8, 16.9, 6.5, 22.1,
          5.3, 10.4, 11.1, 13.9, 11, 20.5),
    fold = c(2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2))

  ivs <- c("x1", "x2")

  fit <- xgb.train(
    verbose = 0,
    params = list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      nthread = n_threads
    ),
    data = xgb.DMatrix(as.matrix(subset(d, fold == 2)[, ivs]), label = subset(d, fold == 2)$y),
    nrounds = 3
  )

  shaps <- as.data.frame(predict(fit,
    newdata = as.matrix(subset(d, fold == 1)[, ivs]),
    predcontrib = TRUE))
  result <- cbind(shaps, sum = rowSums(shaps), pred = predict(fit,
      newdata = as.matrix(subset(d, fold == 1)[, ivs])))

  expect_true(identical(TRUE, all.equal(result$sum, result$pred, tol = 1e-6)))
})


test_that("multiclass feature interactions work", {
  dm <- xgb.DMatrix(
    as.matrix(iris[, -5]), label = as.numeric(iris$Species) - 1, nthread = n_threads
  )
  param <- xgb.params(
    eta = 0.1, max_depth = 4, objective = 'multi:softprob', num_class = 3, nthread = n_threads
  )
  b <- xgb.train(param, dm, 40)
  pred <- predict(b, dm, outputmargin = TRUE)

  # SHAP contributions:
  cont <- predict(b, dm, predcontrib = TRUE)
  expect_length(dim(cont), 3)

  # make sure for each row they add up to marginal predictions
  expect_lt(max(abs(apply(cont, c(1, 2), sum) - pred)), 0.001)

  # SHAP interaction contributions:
  intr <- predict(b, dm, predinteraction = TRUE)
  expect_length(dim(intr), 4)

  # check the symmetry
  expect_lt(max(abs(aperm(intr, c(1, 2, 4, 3)) - intr)), 0.00001)
  # sums WRT columns must be close to feature contributions
  expect_lt(max(abs(apply(intr, c(1, 2, 3), sum) - cont)), 0.00001)
})


test_that("SHAP single sample works", {
  train <- agaricus.train
  test <- agaricus.test
  booster <- xgb.train(
    data = xgb.DMatrix(train$data, label = train$label),
    nrounds = 4,
    params = xgb.params(
      max_depth = 2,
      objective = "binary:logistic",
      nthread = n_threads
    )
  )

  predt <- predict(
    booster,
    newdata = train$data[1, , drop = FALSE], predcontrib = TRUE
  )
  expect_equal(dim(predt), c(1, dim(train$data)[2] + 1))

  predt <- predict(
    booster,
    newdata = train$data[1, , drop = FALSE], predinteraction = TRUE
  )
  expect_equal(dim(predt), c(1, dim(train$data)[2] + 1, dim(train$data)[2] + 1))
})
