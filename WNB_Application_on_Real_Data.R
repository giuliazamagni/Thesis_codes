############################################
## TRUFFLE-2
## DCA and Weighted Net Benefit (WNB)
## with bootstrap validation of AUC and Brier score
############################################

## ===============================
## 0. SETUP
## ===============================
library(pROC)
library(glmnet)
library(ranger)
library(xgboost)
library(dplyr)
library(ggplot2)
library(haven)

set.seed(123)

## ===============================
## 1. LOAD AND PREPROCESS DATA
## ===============================
data <- read_dta("truffle_data_16102024_clean.dta")
data <- na.omit(data)

data$outcome <- as.numeric(as.character(data$outcome))
data$ethnicity <- as.factor(data$ethnicity)
data$PreviousHypMorb <- as.factor(data$PreviousHypMorb)
data$diabetes_all <- as.factor(data$diabetes_all)

prev <- mean(data$outcome)

## ===============================
## 2. MODEL SPECIFICATION
## ===============================

## Elastic-net logistic regression
fit_glm <- function(data) {
  x <- model.matrix(outcome ~ . - 1, data)
  y <- data$outcome
  
  cv <- cv.glmnet(
    x, y,
    family = "binomial",
    alpha = 0.5
  )
  
  glmnet(
    x, y,
    family = "binomial",
    lambda = cv$lambda.min,
    alpha = 0.5
  )
}

pred_glm <- function(model, newdata) {
  x <- model.matrix(outcome ~ . - 1, newdata)
  as.numeric(predict(model, x, type = "response"))
}

## Random forest
fit_rf <- function(data) {
  data$outcome <- factor(data$outcome, levels = c(0, 1))
  
  ranger(
    outcome ~ .,
    data = data,
    probability = TRUE,
    num.trees = 300,
    mtry = floor((ncol(data) - 1) / 3),
    min.node.size = 80,
    max.depth = 5,
    splitrule = "hellinger",
    respect.unordered.factors = "order"
  )
}

pred_rf <- function(model, newdata) {
  predict(model, newdata)$predictions[, 2]
}

## XGBoost
fit_xgb <- function(data) {
  x <- model.matrix(outcome ~ . - 1, data)
  dtrain <- xgb.DMatrix(x, label = data$outcome)
  
  xgb.train(
    data = dtrain,
    objective = "binary:logistic",
    eval_metric = "logloss",
    nrounds = 200,
    eta = 0.02,
    max_depth = 2,
    min_child_weight = 50,
    subsample = 0.7,
    colsample_bytree = 0.7,
    lambda = 5,
    alpha = 1,
    max_delta_step = 1,
    verbose = 0
  )
}

pred_xgb <- function(model, newdata) {
  x <- model.matrix(outcome ~ . - 1, newdata)
  predict(model, xgb.DMatrix(x))
}

models <- list(
  Logistic = list(fit = fit_glm, pred = pred_glm),
  RandomForest = list(fit = fit_rf, pred = pred_rf),
  XGBoost = list(fit = fit_xgb, pred = pred_xgb)
)

## ===============================
## 3. PERFORMANCE METRICS
## ===============================

compute_auc <- function(y, p) {
  as.numeric(roc(y, p, quiet = TRUE)$auc)
}

compute_brier <- function(y, p) {
  mean((p - y)^2)
}

compute_nb <- function(y, p, t) {
  tp <- mean(p >= t & y == 1)
  fp <- mean(p >= t & y == 0)
  
  tp - fp * t / (1 - t)
}

compute_WNB <- function(NB, LCE, wSD) {
  NB / (1 + LCE + wSD)
}

## ===============================
## 4. LOCAL CALIBRATION AND POSTERIOR UNCERTAINTY
## ===============================

compute_LCE_wSD <- function(
    y,
    p,
    t,
    width = 0.05,
    prior_alpha = 3,
    prior_beta = 17
) {
  
  ## Observations within ±0.05 of the decision threshold
  idx <- which(p >= t - width & p < t + width)
  
  if (length(idx) == 0) {
    return(list(LCE = 0, wSD = 0))
  }
  
  k <- sum(y[idx])
  n <- length(idx)
  pbar <- mean(p[idx])
  
  ## Beta-binomial posterior for the local event rate
  a <- prior_alpha + k
  b <- prior_beta + (n - k)
  
  post_mean <- a / (a + b)
  post_sd <- sqrt(
    a * b / ((a + b)^2 * (a + b + 1))
  )
  
  list(
    LCE = abs(pbar - post_mean),
    wSD = post_sd
  )
}

## ===============================
## 5. BOOTSTRAP VALIDATION OF AUC AND BRIER SCORE
## ===============================

bootstrap_auc_brier <- function(data, fit, pred, B = 100) {
  
  y <- data$outcome
  m0 <- fit(data)
  p0 <- pred(m0, data)
  
  auc_app <- compute_auc(y, p0)
  brier_app <- compute_brier(y, p0)
  
  auc_corr <- numeric(B)
  brier_corr <- numeric(B)
  
  for (b in seq_len(B)) {
    
    idx <- sample(
      seq_len(nrow(data)),
      replace = TRUE
    )
    
    db <- data[idx, ]
    mb <- fit(db)
    
    p_db <- pred(mb, db)
    p_orig <- pred(mb, data)
    
    auc_corr[b] <- auc_app -
      (compute_auc(db$outcome, p_db) -
         compute_auc(y, p_orig))
    
    brier_corr[b] <- brier_app -
      (compute_brier(db$outcome, p_db) -
         compute_brier(y, p_orig))
  }
  
  tibble(
    Metric = c("AUC", "Brier"),
    Estimate = c(
      mean(auc_corr),
      mean(brier_corr)
    ),
    LCL = c(
      quantile(auc_corr, 0.025),
      quantile(brier_corr, 0.025)
    ),
    UCL = c(
      quantile(auc_corr, 0.975),
      quantile(brier_corr, 0.975)
    )
  )
}

## ===============================
## 6. RUN ANALYSES
## ===============================

final_table <- lapply(names(models), function(nm) {
  
  fit <- models[[nm]]$fit
  pred <- models[[nm]]$pred
  
  m <- fit(data)
  p <- pred(m, data)
  
  NB <- compute_nb(
    data$outcome,
    p,
    t = 0.20
  )
  
  cal <- compute_LCE_wSD(
    data$outcome,
    p,
    t = 0.20
  )
  
  WNB <- compute_WNB(
    NB,
    cal$LCE,
    cal$wSD
  )
  
  boot_perf <- bootstrap_auc_brier(
    data,
    fit,
    pred
  )
  
  bind_rows(
    boot_perf,
    tibble(
      Metric = c("NB (t=0.20)", "WNB (t=0.20)"),
      Estimate = c(NB, WNB),
      LCL = NA_real_,
      UCL = NA_real_
    )
  ) |>
    mutate(Model = nm)
  
}) |>
  bind_rows() |>
  select(Model, Metric, Estimate, LCL, UCL)

print(final_table)

## ===============================
## 7. CALIBRATION CURVES
## ===============================

calibration_data <- lapply(names(models), function(nm) {
  
  fit <- models[[nm]]$fit
  pred <- models[[nm]]$pred
  
  m <- fit(data)
  p <- pred(m, data)
  
  tibble(
    outcome = data$outcome,
    p = p
  ) |>
    mutate(
      decile = ntile(p, 10)
    ) |>
    group_by(decile) |>
    summarise(
      mean_pred = mean(p),
      obs_rate = mean(outcome),
      n = n(),
      .groups = "drop"
    ) |>
    mutate(Model = nm)
  
}) |>
  bind_rows()

ggplot(
  calibration_data,
  aes(
    x = mean_pred,
    y = obs_rate,
    color = Model
  )
) +
  geom_line(linewidth = 1.1) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "grey40"
  ) +
  coord_equal(
    xlim = c(0, 0.6),
    ylim = c(0, 0.6)
  ) +
  theme_minimal(base_size = 14) +
  labs(
    x = "Mean predicted probability",
    y = "Observed event rate",
    color = ""
  )

## ===============================
## 8. NB, WNB, LCE, AND POSTERIOR SD BY THRESHOLD
## ===============================

thresholds <- seq(
  0.10,
  0.50,
  by = 0.10
)

nb_table_thresholds <- lapply(names(models), function(nm) {
  
  fit <- models[[nm]]$fit
  pred <- models[[nm]]$pred
  
  m <- fit(data)
  p <- pred(m, data)
  y <- data$outcome
  
  lapply(thresholds, function(t) {
    
    NB <- compute_nb(y, p, t)
    cal <- compute_LCE_wSD(y, p, t)
    WNB <- compute_WNB(NB, cal$LCE, cal$wSD)
    
    tibble(
      Model = nm,
      Threshold = t,
      NB = NB,
      WNB = WNB,
      LCE = cal$LCE,
      wSD = cal$wSD,
      N_bin = sum(
        p >= t - 0.05 &
          p < t + 0.05
      )
    )
    
  }) |>
    bind_rows()
  
}) |>
  bind_rows()

print(nb_table_thresholds)