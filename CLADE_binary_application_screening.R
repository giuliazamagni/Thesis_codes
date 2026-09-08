################################################################################
## CLADE MAIN ANALYSIS — AUC / SCREENING ORIENTED
##
## Models:
##   1. AUC-oriented elastic-net logistic regression
##   2. AUC-oriented Random Forest
##   3. AUC-oriented XGBoost
##
## ONE COMMON ORDINARY (NON-STRATIFIED) BOOTSTRAP is used for:
##   - optimism-corrected performance
##   - individual prediction uncertainty
##   - calibration instability
##   - decision fragility
##   - optimism-corrected decision curve analysis
##   - target-sensitivity operating-point analysis
##
## The SAME bootstrap samples are used across ALL models.
## Within each model, each bootstrap model is fitted ONCE and reused for all
## analyses above. This mirrors the most recent PROB-oriented pipeline.
################################################################################

################################################################################
## 0) SETUP
################################################################################

library(haven)
library(dplyr)
library(purrr)
library(pROC)
library(glmnet)
library(ranger)
library(xgboost)
library(ggplot2)
library(tibble)
library(tidyr)
library(writexl)
library(ggpubr)

set.seed(123)

################################################################################
## 1) LOAD AND PREPROCESS DATA
################################################################################

df <- read_dta("Truffle_ready.dta") |>
  as.data.frame() |>
  na.omit()

df$Adverse_outcome <- as.numeric(as.character(df$Adverse_outcome))
df$Caucasian        <- factor(df$Caucasian)
df$Diabetes_cat     <- factor(df$Diabetes_cat)
df$Smoking          <- factor(df$Smoking)

X <- df |>
  dplyr::select(-Adverse_outcome)

y <- df$Adverse_outcome

if (!all(y %in% c(0, 1))) {
  stop("Adverse_outcome must be coded as 0/1.")
}

n <- nrow(df)
prev <- mean(y)

cat("Sample size:", n, "\n")
cat("Events:", sum(y), "\n")
cat("Outcome prevalence:", round(prev, 4), "\n")

################################################################################
## 2) ANALYSIS SETTINGS
################################################################################

## Bootstrap replicates for final analysis
B <- 1000

## Decision-curve threshold range
thresholds_dca <- round(seq(0.01, 0.99, by = 0.01), 2)

## Prespecified thresholds for fragility and reporting
thresholds_interest <- c(0.05, 0.08, 0.10, 0.15)

## Target screening sensitivity
target_sens <- 0.95

## Number of permutation-importance repeats
n_perm_importance <- 5

################################################################################
## 3) HELPER FUNCTIONS
################################################################################

clamp01 <- function(p) {
  pmin(pmax(p, 1e-15), 1 - 1e-15)
}

safe_sd <- function(x) {
  if (sum(!is.na(x)) <= 1) return(NA_real_)
  stats::sd(x, na.rm = TRUE)
}

safe_q <- function(x, prob) {
  if (all(is.na(x))) return(NA_real_)
  as.numeric(
    stats::quantile(
      x,
      probs = prob,
      na.rm = TRUE,
      type = 7
    )
  )
}

make_progress <- function(total) {
  utils::txtProgressBar(
    min = 0,
    max = total,
    style = 3,
    width = 50,
    char = "="
  )
}

################################################################################
## 4) PERFORMANCE METRICS
################################################################################

AUC_fun <- function(y, p) {
  if (length(unique(y)) < 2) return(NA_real_)

  as.numeric(
    pROC::roc(
      response = y,
      predictor = p,
      quiet = TRUE
    )$auc
  )
}

Brier_fun <- function(y, p) {
  mean((p - y)^2, na.rm = TRUE)
}

LogLoss_fun <- function(y, p) {
  p <- clamp01(p)

  -mean(
    y * log(p) +
      (1 - y) * log(1 - p),
    na.rm = TRUE
  )
}

CalSlope <- function(y, p) {
  p <- clamp01(p)
  lp <- qlogis(p)

  if (length(unique(y)) < 2) return(NA_real_)
  if (any(!is.finite(lp))) return(NA_real_)

  fit <- try(
    stats::glm(
      y ~ lp,
      family = stats::binomial()
    ),
    silent = TRUE
  )

  if (inherits(fit, "try-error")) return(NA_real_)

  co <- try(stats::coef(fit), silent = TRUE)

  if (inherits(co, "try-error")) return(NA_real_)
  if (length(co) < 2 || !is.finite(co[2])) return(NA_real_)

  unname(co[2])
}

CalIntercept <- function(y, p) {
  p <- clamp01(p)
  lp <- qlogis(p)

  if (length(unique(y)) < 2) return(NA_real_)
  if (any(!is.finite(lp))) return(NA_real_)

  fit <- try(
    stats::glm(
      y ~ offset(lp),
      family = stats::binomial()
    ),
    silent = TRUE
  )

  if (inherits(fit, "try-error")) return(NA_real_)

  co <- try(stats::coef(fit), silent = TRUE)

  if (inherits(co, "try-error")) return(NA_real_)
  if (length(co) < 1 || !is.finite(co[1])) return(NA_real_)

  unname(co[1])
}

metric_vec <- function(y, p) {
  tibble::tibble(
    AUC = AUC_fun(y, p),
    Brier = Brier_fun(y, p),
    CalIntercept = CalIntercept(y, p),
    CalSlope = CalSlope(y, p)
  )
}

################################################################################
## 5) TARGET-SENSITIVITY OPERATING POINT
################################################################################

bin_metrics_at_threshold <- function(y, p, threshold) {
  pred <- ifelse(p >= threshold, 1, 0)

  TP <- sum(pred == 1 & y == 1, na.rm = TRUE)
  FP <- sum(pred == 1 & y == 0, na.rm = TRUE)
  TN <- sum(pred == 0 & y == 0, na.rm = TRUE)
  FN <- sum(pred == 0 & y == 1, na.rm = TRUE)

  sensitivity <- if ((TP + FN) == 0) NA_real_ else TP / (TP + FN)
  specificity <- if ((TN + FP) == 0) NA_real_ else TN / (TN + FP)
  ppv <- if ((TP + FP) == 0) NA_real_ else TP / (TP + FP)
  npv <- if ((TN + FN) == 0) NA_real_ else TN / (TN + FN)

  tibble::tibble(
    threshold = threshold,
    sensitivity = sensitivity,
    specificity = specificity,
    PPV = ppv,
    NPV = npv,
    TP = TP,
    FP = FP,
    TN = TN,
    FN = FN
  )
}

find_threshold_for_target_sensitivity <- function(y, p, target_sens = 0.95) {
  p <- clamp01(p)

  if (length(unique(y)) < 2) {
    stop("Both outcome classes are required to select a sensitivity threshold.")
  }

  cand <- sort(unique(p), decreasing = TRUE)

  sens_vec <- vapply(
    cand,
    function(thr) {
      pred <- ifelse(p >= thr, 1, 0)
      TP <- sum(pred == 1 & y == 1, na.rm = TRUE)
      FN <- sum(pred == 0 & y == 1, na.rm = TRUE)

      if ((TP + FN) == 0) return(NA_real_)
      TP / (TP + FN)
    },
    numeric(1)
  )

  ok <- which(!is.na(sens_vec) & sens_vec >= target_sens)

  if (length(ok) == 0) {
    idx_best <- which.min(abs(sens_vec - target_sens))
  } else {
    ## Highest threshold that still achieves the target sensitivity
    idx_best <- ok[1]
  }

  tibble::tibble(
    threshold = cand[idx_best],
    achieved_sensitivity = sens_vec[idx_best]
  )
}

################################################################################
## 6) SHARED ORDINARY BOOTSTRAP
##
## The prevalence is allowed to vary naturally across bootstrap samples.
################################################################################

set.seed(123)

bootstrap_indices <- replicate(
  B,
  sample(
    seq_len(n),
    size = n,
    replace = TRUE
  ),
  simplify = FALSE
)

bootstrap_prevalence <- purrr::map_dbl(
  bootstrap_indices,
  ~ mean(y[.x])
)

cat(
  "Bootstrap prevalence range:",
  round(min(bootstrap_prevalence), 4),
  "-",
  round(max(bootstrap_prevalence), 4),
  "\n"
)

################################################################################
## 7) MODEL MATRIX
################################################################################

make_mm <- function(newdata, template_cols) {
  mm <- stats::model.matrix(
    ~ . - 1,
    data = newdata
  )

  missing_cols <- setdiff(template_cols, colnames(mm))

  if (length(missing_cols) > 0) {
    mm_missing <- matrix(
      0,
      nrow = nrow(mm),
      ncol = length(missing_cols)
    )

    colnames(mm_missing) <- missing_cols
    mm <- cbind(mm, mm_missing)
  }

  mm <- mm[, template_cols, drop = FALSE]
  mm
}

X_full <- stats::model.matrix(~ . - 1, X)
mm_cols <- colnames(X_full)

################################################################################
## 8) AUC-ORIENTED MODEL FIT AND PREDICTION FUNCTIONS
################################################################################

############################
## 8A) LOGISTIC ELASTIC-NET
############################

fit_logit <- function(idx, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  x_boot <- X_full[idx, , drop = FALSE]
  y_boot <- y[idx]

  if (length(unique(y_boot)) < 2) {
    stop("Only one outcome class present.")
  }

  prev_boot <- mean(y_boot)

  if (!is.finite(prev_boot) || prev_boot <= 0 || prev_boot >= 1) {
    stop("Bootstrap sample has invalid event prevalence for weighted logistic fit.")
  }

  ## Class weighting for screening / discrimination-oriented development
  w_boot <- ifelse(
    y_boot == 1,
    1 / prev_boot,
    1 / (1 - prev_boot)
  )

  cv_logit <- glmnet::cv.glmnet(
    x = x_boot,
    y = y_boot,
    weights = w_boot,
    family = "binomial",
    alpha = 0.5,
    type.measure = "auc"
  )

  glmnet::glmnet(
    x = x_boot,
    y = y_boot,
    weights = w_boot,
    family = "binomial",
    alpha = 0.5,
    lambda = cv_logit$lambda.min
  )
}

pred_logit <- function(model, newdata) {
  mm_new <- make_mm(newdata, mm_cols)

  as.numeric(
    stats::predict(
      model,
      newx = mm_new,
      type = "response"
    )
  )
}

############################
## 8B) RANDOM FOREST
############################

fit_rf <- function(idx, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  df_boot <- df[idx, , drop = FALSE]

  if (length(unique(df_boot$Adverse_outcome)) < 2) {
    stop("Only one outcome class present.")
  }

  df_boot$Adverse_outcome <- factor(
    df_boot$Adverse_outcome,
    levels = c(0, 1)
  )

  ranger::ranger(
    Adverse_outcome ~ .,
    data = df_boot,
    probability = TRUE,
    num.trees = 800,
    mtry = 3,
    min.node.size = 10,
    max.depth = 5,
    sample.fraction = 0.7,
    splitrule = "hellinger",
    respect.unordered.factors = "order",
    seed = seed
  )
}

pred_rf <- function(model, newdata) {
  pr <- stats::predict(
    model,
    data = newdata
  )$predictions

  if ("1" %in% colnames(pr)) {
    return(as.numeric(pr[, "1"]))
  }

  as.numeric(pr[, ncol(pr)])
}

############################
## 8C) XGBOOST
############################

fit_xgb <- function(idx, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  x_boot <- X_full[idx, , drop = FALSE]
  y_boot <- y[idx]

  n_pos <- sum(y_boot == 1)
  n_neg <- sum(y_boot == 0)

  if (n_pos == 0 || n_neg == 0) {
    stop("Bootstrap sample has only one outcome class for XGBoost.")
  }

  pos_weight <- n_neg / n_pos

  dtrain <- xgboost::xgb.DMatrix(
    data = x_boot,
    label = y_boot
  )

  cv <- xgboost::xgb.cv(
    data = dtrain,
    objective = "binary:logistic",
    eval_metric = "auc",
    nrounds = 800,
    eta = 0.05,
    max_depth = 3,
    min_child_weight = 20,
    subsample = 0.8,
    colsample_bytree = 0.8,
    lambda = 5,
    alpha = 1,
    scale_pos_weight = pos_weight,
    max_delta_step = 1,
    nfold = 5,
    early_stopping_rounds = 50,
    verbose = 0
  )

  best_nrounds <- cv$best_iteration

  if (is.null(best_nrounds) || is.na(best_nrounds) || best_nrounds < 1) {
    best_nrounds <- 50
  }

  xgboost::xgb.train(
    data = dtrain,
    nrounds = best_nrounds,
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.05,
    max_depth = 3,
    min_child_weight = 20,
    subsample = 0.8,
    colsample_bytree = 0.8,
    lambda = 5,
    alpha = 1,
    scale_pos_weight = pos_weight,
    max_delta_step = 1,
    verbose = 0
  )
}

pred_xgb <- function(model, newdata) {
  mm_new <- make_mm(newdata, mm_cols)

  as.numeric(
    stats::predict(
      model,
      xgboost::xgb.DMatrix(mm_new)
    )
  )
}

################################################################################
## 9) MODEL LIST
################################################################################

models <- list(
  Logistic = list(
    fit = fit_logit,
    pred = pred_logit
  ),
  RandomForest = list(
    fit = fit_rf,
    pred = pred_rf
  ),
  XGBoost = list(
    fit = fit_xgb,
    pred = pred_xgb
  )
)

################################################################################
## 10) DECISION CURVE ANALYSIS
################################################################################

decision_curve_data <- function(
    y,
    p,
    thresholds = thresholds_dca) {

  thresholds <- round(thresholds, 2)
  n_local <- length(y)
  prevalence <- mean(y, na.rm = TRUE)

  tibble::tibble(
    threshold = thresholds
  ) |>
    dplyr::mutate(
      TP = purrr::map_dbl(
        threshold,
        ~ sum(p >= .x & y == 1, na.rm = TRUE)
      ),
      FP = purrr::map_dbl(
        threshold,
        ~ sum(p >= .x & y == 0, na.rm = TRUE)
      ),
      NB_model =
        (TP / n_local) -
        (FP / n_local) *
        threshold / (1 - threshold),
      NB_all =
        prevalence -
        (1 - prevalence) *
        threshold / (1 - threshold),
      NB_none = 0
    )
}

################################################################################
## 11) PATIENT-LEVEL BOOTSTRAP UNCERTAINTY
################################################################################

summarise_patient_bootstrap <- function(
    pred_boot,
    apparent_pred,
    y,
    model_name) {

  tibble::tibble(
    patient_id = seq_len(nrow(pred_boot)),
    outcome = y,
    pred_apparent = apparent_pred,
    pred_boot_mean = rowMeans(pred_boot, na.rm = TRUE),
    pred_boot_sd = apply(pred_boot, 1, safe_sd),
    pred_boot_p025 = apply(pred_boot, 1, safe_q, prob = 0.025),
    pred_boot_p975 = apply(pred_boot, 1, safe_q, prob = 0.975)
  ) |>
    dplyr::mutate(
      pred_boot_interval_width =
        pred_boot_p975 - pred_boot_p025,
      model = model_name
    )
}

################################################################################
## 12) DECISION FRAGILITY
################################################################################

compute_fragility <- function(
    patient_tbl,
    thresholds,
    model_name) {

  purrr::map_dfr(
    thresholds,
    function(thr) {
      fragile <-
        patient_tbl$pred_boot_p025 <= thr &
        patient_tbl$pred_boot_p975 >= thr

      tibble::tibble(
        Model = model_name,
        threshold = thr,
        n_fragile = sum(fragile, na.rm = TRUE),
        n_total = sum(!is.na(fragile)),
        prop_fragile = mean(fragile, na.rm = TRUE),
        n_above = sum(patient_tbl$pred_boot_p025 > thr, na.rm = TRUE),
        n_below = sum(patient_tbl$pred_boot_p975 < thr, na.rm = TRUE)
      )
    }
  )
}

################################################################################
## 13) CALIBRATION INSTABILITY
################################################################################

make_breaks_deciles <- function(p, g = 10) {
  qs <- unique(
    stats::quantile(
      p,
      probs = seq(0, 1, length.out = g + 1),
      na.rm = TRUE
    )
  )

  if (length(qs) < 3) {
    qs <- unique(
      c(
        min(p, na.rm = TRUE),
        median(p, na.rm = TRUE),
        max(p, na.rm = TRUE)
      )
    )
  }

  qs[1] <- -Inf
  qs[length(qs)] <- Inf
  qs
}

assign_groups <- function(p, breaks) {
  as.integer(
    cut(
      p,
      breaks = breaks,
      include.lowest = TRUE,
      labels = FALSE
    )
  )
}

calibration_instability_data <- function(
    apparent_pred,
    pred_boot,
    y,
    g = 10) {

  breaks <- make_breaks_deciles(apparent_pred, g)

  apparent_df <-
    tibble::tibble(
      y = y,
      p = apparent_pred
    ) |>
    dplyr::mutate(
      group = assign_groups(p, breaks)
    ) |>
    dplyr::group_by(group) |>
    dplyr::summarise(
      mean_pred = mean(p, na.rm = TRUE),
      obs = mean(y, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      Type = "Apparent",
      bootstrap = NA_integer_
    )

  boot_df <- purrr::map_dfr(
    seq_len(ncol(pred_boot)),
    function(j) {
      tibble::tibble(
        y = y,
        p = pred_boot[, j]
      ) |>
        dplyr::mutate(
          group = assign_groups(p, breaks)
        ) |>
        dplyr::group_by(group) |>
        dplyr::summarise(
          mean_pred = mean(p, na.rm = TRUE),
          obs = mean(y, na.rm = TRUE),
          .groups = "drop"
        ) |>
        dplyr::mutate(
          Type = "Bootstrap",
          bootstrap = j
        )
    }
  )

  dplyr::bind_rows(apparent_df, boot_df)
}

################################################################################
## 14) AUC-BASED PERMUTATION IMPORTANCE
################################################################################

permutation_importance_auc <- function(
    model,
    pred_fun,
    X,
    y,
    model_name,
    n_perm = 5) {

  cat("Permutation importance (AUC):", model_name, "\n")

  p_ref <- clamp01(pred_fun(model, X))
  base_auc <- AUC_fun(y, p_ref)

  vars <- colnames(X)
  pb <- make_progress(length(vars))

  on.exit(
    try(close(pb), silent = TRUE),
    add = TRUE
  )

  out <- purrr::map_dfr(
    seq_along(vars),
    function(j) {
      utils::setTxtProgressBar(pb, j)

      variable <- vars[j]
      decreases <- rep(NA_real_, n_perm)

      for (k in seq_len(n_perm)) {
        X_perm <- X
        X_perm[[variable]] <- sample(
          X_perm[[variable]],
          replace = FALSE
        )

        p_perm <- try(
          clamp01(pred_fun(model, X_perm)),
          silent = TRUE
        )

        if (inherits(p_perm, "try-error")) next

        decreases[k] <-
          base_auc - AUC_fun(y, p_perm)
      }

      tibble::tibble(
        Model = model_name,
        Feature = variable,
        base_auc = base_auc,
        mean_decrease_auc = mean(decreases, na.rm = TRUE),
        sd_decrease_auc = stats::sd(decreases, na.rm = TRUE),
        q025_decrease_auc = safe_q(decreases, 0.025),
        q975_decrease_auc = safe_q(decreases, 0.975),
        n_valid_perm = sum(!is.na(decreases))
      )
    }
  )

  cat("\n")

  out |>
    dplyr::arrange(
      dplyr::desc(mean_decrease_auc)
    )
}

################################################################################
## 15) SHARED-BOOTSTRAP PIPELINE FOR ONE MODEL
##
## Each bootstrap model is FIT ONCE.
## The same fit is reused for:
##   - optimism correction of AUC/Brier/calibration
##   - individual uncertainty and fragility
##   - DCA optimism correction
##   - target-sensitivity operating-point optimism correction
################################################################################

run_model_pipeline <- function(
    model_name,
    fit_fun,
    pred_fun,
    X,
    y,
    bootstrap_indices,
    thresholds_dca,
    thresholds_interest,
    target_sens = 0.95,
    n_perm_importance = 5,
    base_seed = 10000) {

  cat("\n")
  cat("====================================================\n")
  cat("MODEL:", model_name, "\n")
  cat("====================================================\n")

  ##########################################################################
  ## 15A) APPARENT MODEL
  ##########################################################################

  apparent_model <- fit_fun(
    seq_len(nrow(X)),
    seed = base_seed
  )

  apparent_pred <- clamp01(
    pred_fun(apparent_model, X)
  )

  apparent_metrics <- metric_vec(
    y,
    apparent_pred
  )

  apparent_dca <-
    decision_curve_data(
      y = y,
      p = apparent_pred,
      thresholds = thresholds_dca
    ) |>
    dplyr::transmute(
      threshold,
      NB_apparent = NB_model,
      NB_all,
      NB_none
    )

  apparent_target_threshold <-
    find_threshold_for_target_sensitivity(
      y = y,
      p = apparent_pred,
      target_sens = target_sens
    )

  apparent_target_metrics <-
    bin_metrics_at_threshold(
      y = y,
      p = apparent_pred,
      threshold = apparent_target_threshold$threshold
    ) |>
    dplyr::mutate(
      target_sensitivity = target_sens,
      achieved_sensitivity = sensitivity
    )

  ##########################################################################
  ## 15B) STORAGE
  ##########################################################################

  B_local <- length(bootstrap_indices)

  pred_boot <- matrix(
    NA_real_,
    nrow = nrow(X),
    ncol = B_local
  )

  performance_list <- vector("list", B_local)
  dca_list <- vector("list", B_local)
  target_sens_list <- vector("list", B_local)

  valid <- logical(B_local)

  fail_log <- tibble::tibble(
    bootstrap = integer(),
    stage = character(),
    message = character()
  )

  ##########################################################################
  ## 15C) ONE BOOTSTRAP LOOP
  ##########################################################################

  pb <- make_progress(B_local)

  on.exit(
    try(close(pb), silent = TRUE),
    add = TRUE
  )

  for (b in seq_len(B_local)) {
    utils::setTxtProgressBar(pb, b)

    idxb <- bootstrap_indices[[b]]

    ########################################################################
    ## FIT MODEL ONCE
    ########################################################################

    model_b <- try(
      fit_fun(
        idxb,
        seed = base_seed + b
      ),
      silent = TRUE
    )

    if (inherits(model_b, "try-error")) {
      fail_log <- dplyr::bind_rows(
        fail_log,
        tibble::tibble(
          bootstrap = b,
          stage = "fit",
          message = as.character(model_b)
        )
      )
      next
    }

    ########################################################################
    ## PREDICT ON ORIGINAL DATA ONCE
    ##
    ## Used for:
    ##   - test performance
    ##   - individual uncertainty / fragility
    ##   - test DCA
    ##   - target-sensitivity test performance
    ########################################################################

    p_original <- try(
      clamp01(
        pred_fun(model_b, X)
      ),
      silent = TRUE
    )

    if (
      inherits(p_original, "try-error") ||
      any(!is.finite(p_original))
    ) {
      fail_log <- dplyr::bind_rows(
        fail_log,
        tibble::tibble(
          bootstrap = b,
          stage = "predict_original",
          message = "Prediction failed or returned non-finite values."
        )
      )
      next
    }

    ########################################################################
    ## PREDICTIONS WITHIN BOOTSTRAP SAMPLE
    ##
    ## Because bootstrap subjects are rows of the original dataset,
    ## p_original[idxb] is the prediction for the bootstrap sample and avoids
    ## a second predict() call.
    ########################################################################

    p_bootsample <- p_original[idxb]

    ########################################################################
    ## STORE PATIENT-LEVEL PREDICTIONS
    ########################################################################

    pred_boot[, b] <- p_original

    ########################################################################
    ## PERFORMANCE
    ########################################################################

    perf_boot <- metric_vec(
      y[idxb],
      p_bootsample
    )

    perf_test <- metric_vec(
      y,
      p_original
    )

    performance_list[[b]] <-
      dplyr::bind_cols(
        tibble::tibble(
          bootstrap = b,
          bootstrap_prevalence = mean(y[idxb])
        ),
        perf_boot |>
          dplyr::rename_with(
            ~ paste0(.x, "_boot")
          ),
        perf_test |>
          dplyr::rename_with(
            ~ paste0(.x, "_test")
          )
      )

    ########################################################################
    ## DCA
    ########################################################################

    dca_boot <-
      decision_curve_data(
        y = y[idxb],
        p = p_bootsample,
        thresholds = thresholds_dca
      ) |>
      dplyr::select(
        threshold,
        NB_boot = NB_model
      )

    dca_test <-
      decision_curve_data(
        y = y,
        p = p_original,
        thresholds = thresholds_dca
      ) |>
      dplyr::select(
        threshold,
        NB_test = NB_model
      )

    dca_list[[b]] <-
      dca_boot |>
      dplyr::left_join(
        dca_test,
        by = "threshold"
      ) |>
      dplyr::mutate(
        bootstrap = b,
        NB_optimism = NB_boot - NB_test,
        .before = 1
      )

    ########################################################################
    ## TARGET-SENSITIVITY OPERATING POINT
    ##
    ## Select the threshold in the bootstrap sample, then evaluate THAT SAME
    ## threshold on the original sample using the same bootstrap-fitted model.
    ########################################################################

    threshold_b <- try(
      find_threshold_for_target_sensitivity(
        y = y[idxb],
        p = p_bootsample,
        target_sens = target_sens
      ),
      silent = TRUE
    )

    if (inherits(threshold_b, "try-error")) {
      fail_log <- dplyr::bind_rows(
        fail_log,
        tibble::tibble(
          bootstrap = b,
          stage = "target_sensitivity_threshold",
          message = as.character(threshold_b)
        )
      )
      next
    }

    target_boot <- bin_metrics_at_threshold(
      y = y[idxb],
      p = p_bootsample,
      threshold = threshold_b$threshold
    )

    target_test <- bin_metrics_at_threshold(
      y = y,
      p = p_original,
      threshold = threshold_b$threshold
    )

    target_sens_list[[b]] <- tibble::tibble(
      bootstrap = b,
      threshold_boot = threshold_b$threshold,
      sensitivity_boot = target_boot$sensitivity,
      specificity_boot = target_boot$specificity,
      PPV_boot = target_boot$PPV,
      NPV_boot = target_boot$NPV,
      sensitivity_test = target_test$sensitivity,
      specificity_test = target_test$specificity,
      PPV_test = target_test$PPV,
      NPV_test = target_test$NPV
    )

    valid[b] <- TRUE
  }

  cat("\n")

  ##########################################################################
  ## 15D) KEEP VALID REPLICATIONS
  ##########################################################################

  if (sum(valid) < 2) {
    stop(
      paste(
        "Too few valid bootstrap replications for",
        model_name
      )
    )
  }

  pred_boot <- pred_boot[, valid, drop = FALSE]
  valid_bootstrap_ids <- which(valid)

  colnames(pred_boot) <- paste0(
    "boot_",
    valid_bootstrap_ids
  )

  performance_boot <- dplyr::bind_rows(performance_list)
  dca_boot_detail <- dplyr::bind_rows(dca_list)
  target_sens_boot <- dplyr::bind_rows(target_sens_list)

  cat(
    "Valid bootstrap replications:",
    sum(valid),
    "/",
    B_local,
    "\n"
  )

  ##########################################################################
  ## 15E) OPTIMISM-CORRECTED PERFORMANCE
  ##########################################################################

  performance_optimism <-
    performance_boot |>
    dplyr::transmute(
      bootstrap,
      AUC_optimism = AUC_boot - AUC_test,
      Brier_optimism = Brier_boot - Brier_test,
      CalIntercept_optimism = CalIntercept_boot - CalIntercept_test,
      CalSlope_optimism = CalSlope_boot - CalSlope_test
    )

  performance_corrected_dist <-
    performance_optimism |>
    dplyr::transmute(
      bootstrap,
      AUC_corrected_iter =
        apparent_metrics$AUC - AUC_optimism,
      Brier_corrected_iter =
        apparent_metrics$Brier - Brier_optimism,
      CalIntercept_corrected_iter =
        apparent_metrics$CalIntercept - CalIntercept_optimism,
      CalSlope_corrected_iter =
        apparent_metrics$CalSlope - CalSlope_optimism
    )

  performance_summary <- tibble::tibble(
    Model = model_name,

    AUC_apparent = apparent_metrics$AUC,
    AUC_corrected =
      apparent_metrics$AUC -
      mean(performance_optimism$AUC_optimism, na.rm = TRUE),
    AUC_p025 = safe_q(
      performance_corrected_dist$AUC_corrected_iter,
      0.025
    ),
    AUC_p975 = safe_q(
      performance_corrected_dist$AUC_corrected_iter,
      0.975
    ),

    Brier_apparent = apparent_metrics$Brier,
    Brier_corrected =
      apparent_metrics$Brier -
      mean(performance_optimism$Brier_optimism, na.rm = TRUE),
    Brier_p025 = safe_q(
      performance_corrected_dist$Brier_corrected_iter,
      0.025
    ),
    Brier_p975 = safe_q(
      performance_corrected_dist$Brier_corrected_iter,
      0.975
    ),

    CalIntercept_apparent = apparent_metrics$CalIntercept,
    CalIntercept_corrected =
      apparent_metrics$CalIntercept -
      mean(
        performance_optimism$CalIntercept_optimism,
        na.rm = TRUE
      ),
    CalIntercept_p025 = safe_q(
      performance_corrected_dist$CalIntercept_corrected_iter,
      0.025
    ),
    CalIntercept_p975 = safe_q(
      performance_corrected_dist$CalIntercept_corrected_iter,
      0.975
    ),

    CalSlope_apparent = apparent_metrics$CalSlope,
    CalSlope_corrected =
      apparent_metrics$CalSlope -
      mean(
        performance_optimism$CalSlope_optimism,
        na.rm = TRUE
      ),
    CalSlope_p025 = safe_q(
      performance_corrected_dist$CalSlope_corrected_iter,
      0.025
    ),
    CalSlope_p975 = safe_q(
      performance_corrected_dist$CalSlope_corrected_iter,
      0.975
    ),

    n_valid_bootstrap = sum(valid)
  )

  ##########################################################################
  ## 15F) PATIENT-LEVEL UNCERTAINTY
  ##########################################################################

  patient_uncertainty <-
    summarise_patient_bootstrap(
      pred_boot = pred_boot,
      apparent_pred = apparent_pred,
      y = y,
      model_name = model_name
    )

  ##########################################################################
  ## 15G) DECISION FRAGILITY
  ##########################################################################

  fragility <-
    compute_fragility(
      patient_tbl = patient_uncertainty,
      thresholds = thresholds_interest,
      model_name = model_name
    )

  ##########################################################################
  ## 15H) CALIBRATION INSTABILITY
  ##########################################################################

  calibration_instability <-
    calibration_instability_data(
      apparent_pred = apparent_pred,
      pred_boot = pred_boot,
      y = y,
      g = 10
    )

  ##########################################################################
  ## 15I) OPTIMISM-CORRECTED DCA
  ##########################################################################

  dca_optimism <-
    dca_boot_detail |>
    dplyr::group_by(threshold) |>
    dplyr::summarise(
      mean_NB_optimism = mean(NB_optimism, na.rm = TRUE),
      n_valid = sum(!is.na(NB_optimism)),
      .groups = "drop"
    )

  dca_corrected <-
    apparent_dca |>
    dplyr::left_join(
      dca_optimism,
      by = "threshold"
    ) |>
    dplyr::mutate(
      NB_corrected = NB_apparent - mean_NB_optimism
    )

  ##########################################################################
  ## 15J) OPTIMISM-CORRECTED TARGET-SENSITIVITY PERFORMANCE
  ##########################################################################

  target_sens_optimism <-
    target_sens_boot |>
    dplyr::transmute(
      bootstrap,
      sensitivity_optimism =
        sensitivity_boot - sensitivity_test,
      specificity_optimism =
        specificity_boot - specificity_test,
      PPV_optimism =
        PPV_boot - PPV_test,
      NPV_optimism =
        NPV_boot - NPV_test
    )

  target_sens_corrected_dist <-
    target_sens_optimism |>
    dplyr::transmute(
      bootstrap,
      sensitivity_corrected_iter =
        apparent_target_metrics$sensitivity - sensitivity_optimism,
      specificity_corrected_iter =
        apparent_target_metrics$specificity - specificity_optimism,
      PPV_corrected_iter =
        apparent_target_metrics$PPV - PPV_optimism,
      NPV_corrected_iter =
        apparent_target_metrics$NPV - NPV_optimism
    )

  target_sens_summary <- tibble::tibble(
    Model = model_name,
    target_sensitivity = target_sens,
    threshold_apparent = apparent_target_metrics$threshold,

    sensitivity_apparent = apparent_target_metrics$sensitivity,
    sensitivity_corrected =
      apparent_target_metrics$sensitivity -
      mean(
        target_sens_optimism$sensitivity_optimism,
        na.rm = TRUE
      ),
    sensitivity_p025 = safe_q(
      target_sens_corrected_dist$sensitivity_corrected_iter,
      0.025
    ),
    sensitivity_p975 = safe_q(
      target_sens_corrected_dist$sensitivity_corrected_iter,
      0.975
    ),

    specificity_apparent = apparent_target_metrics$specificity,
    specificity_corrected =
      apparent_target_metrics$specificity -
      mean(
        target_sens_optimism$specificity_optimism,
        na.rm = TRUE
      ),
    specificity_p025 = safe_q(
      target_sens_corrected_dist$specificity_corrected_iter,
      0.025
    ),
    specificity_p975 = safe_q(
      target_sens_corrected_dist$specificity_corrected_iter,
      0.975
    ),

    PPV_apparent = apparent_target_metrics$PPV,
    PPV_corrected =
      apparent_target_metrics$PPV -
      mean(
        target_sens_optimism$PPV_optimism,
        na.rm = TRUE
      ),
    PPV_p025 = safe_q(
      target_sens_corrected_dist$PPV_corrected_iter,
      0.025
    ),
    PPV_p975 = safe_q(
      target_sens_corrected_dist$PPV_corrected_iter,
      0.975
    ),

    NPV_apparent = apparent_target_metrics$NPV,
    NPV_corrected =
      apparent_target_metrics$NPV -
      mean(
        target_sens_optimism$NPV_optimism,
        na.rm = TRUE
      ),
    NPV_p025 = safe_q(
      target_sens_corrected_dist$NPV_corrected_iter,
      0.025
    ),
    NPV_p975 = safe_q(
      target_sens_corrected_dist$NPV_corrected_iter,
      0.975
    ),

    n_valid_bootstrap = nrow(target_sens_boot)
  )

  ##########################################################################
  ## 15K) VARIABLE IMPORTANCE
  ##########################################################################

  importance <-
    permutation_importance_auc(
      model = apparent_model,
      pred_fun = pred_fun,
      X = X,
      y = y,
      model_name = model_name,
      n_perm = n_perm_importance
    )

  ##########################################################################
  ## RETURN
  ##########################################################################

  list(
    apparent_model = apparent_model,
    apparent_pred = apparent_pred,

    performance_summary = performance_summary,
    performance_boot = performance_boot,
    performance_optimism = performance_optimism,
    performance_corrected_dist = performance_corrected_dist,

    pred_boot = pred_boot,
    patient_uncertainty = patient_uncertainty,
    fragility = fragility,
    calibration_instability = calibration_instability,

    dca = dca_corrected,
    dca_boot_detail = dca_boot_detail,

    target_sens_summary = target_sens_summary,
    target_sens_apparent = apparent_target_metrics,
    target_sens_boot = target_sens_boot,
    target_sens_optimism = target_sens_optimism,
    target_sens_corrected_dist = target_sens_corrected_dist,

    importance = importance,
    fail_log = fail_log,
    valid_bootstrap_ids = valid_bootstrap_ids
  )
}

################################################################################
## 16) RUN ALL MODELS
################################################################################

all_results <- list()

for (model_name in names(models)) {
  all_results[[model_name]] <-
    run_model_pipeline(
      model_name = model_name,
      fit_fun = models[[model_name]]$fit,
      pred_fun = models[[model_name]]$pred,
      X = X,
      y = y,
      bootstrap_indices = bootstrap_indices,
      thresholds_dca = thresholds_dca,
      thresholds_interest = thresholds_interest,
      target_sens = target_sens,
      n_perm_importance = n_perm_importance,
      base_seed = 10000
    )
}

################################################################################
## 17) PERFORMANCE SUMMARY
################################################################################

performance_summary <-
  dplyr::bind_rows(
    lapply(
      all_results,
      `[[`,
      "performance_summary"
    )
  )

print(performance_summary)

################################################################################
## 18) TARGET-SENSITIVITY SUMMARY
################################################################################

target_sens_summary <-
  dplyr::bind_rows(
    lapply(
      all_results,
      `[[`,
      "target_sens_summary"
    )
  )

print(target_sens_summary)

################################################################################
## 19) PATIENT-LEVEL PREDICTION UNCERTAINTY
################################################################################

patient_uncertainty_all <-
  dplyr::bind_rows(
    lapply(
      all_results,
      `[[`,
      "patient_uncertainty"
    )
  )

uncertainty_summary <-
  patient_uncertainty_all |>
  dplyr::group_by(model) |>
  dplyr::summarise(
    n = dplyr::n(),
    mean_pred_sd = mean(pred_boot_sd, na.rm = TRUE),
    median_pred_sd = median(pred_boot_sd, na.rm = TRUE),
    q90_pred_sd = as.numeric(
      quantile(pred_boot_sd, 0.90, na.rm = TRUE)
    ),
    q95_pred_sd = as.numeric(
      quantile(pred_boot_sd, 0.95, na.rm = TRUE)
    ),
    mean_interval_width = mean(
      pred_boot_interval_width,
      na.rm = TRUE
    ),
    median_interval_width = median(
      pred_boot_interval_width,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) |>
  dplyr::rename(Model = model)

print(uncertainty_summary)

################################################################################
## 20) DECISION FRAGILITY SUMMARY
################################################################################

fragility_all <-
  dplyr::bind_rows(
    lapply(
      all_results,
      `[[`,
      "fragility"
    )
  ) |>
  dplyr::mutate(
    Fragility_percent = 100 * prop_fragile
  )

print(fragility_all)

################################################################################
## 21) NET BENEFIT AT PRESPECIFIED THRESHOLDS
################################################################################

dca_nb_table <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(model_name) {
        all_results[[model_name]]$dca |>
          dplyr::mutate(
            threshold_round = round(threshold, 2)
          ) |>
          dplyr::filter(
            threshold_round %in% thresholds_interest
          ) |>
          dplyr::transmute(
            Model = model_name,
            threshold = threshold_round,
            NB_apparent = NB_apparent,
            NB_corrected = NB_corrected,
            NB_all = NB_all,
            NB_none = NB_none
          )
      }
    )
  )

print(dca_nb_table)

################################################################################
## 22) FINAL NET BENEFIT AND FRAGILITY TABLE
################################################################################

final_table <-
  dca_nb_table |>
  dplyr::left_join(
    fragility_all |>
      dplyr::select(
        Model,
        threshold,
        n_fragile,
        Fragility_percent
      ),
    by = c("Model", "threshold")
  ) |>
  dplyr::arrange(Model, threshold)

print(final_table)

final_wide <-
  final_table |>
  dplyr::select(
    Model,
    threshold,
    NB_corrected,
    Fragility_percent
  ) |>
  tidyr::pivot_wider(
    names_from = threshold,
    values_from = c(
      NB_corrected,
      Fragility_percent
    ),
    names_glue = "t={threshold}_{.value}"
  )

print(final_wide)

################################################################################
## 23) CALIBRATION INSTABILITY PLOTS
################################################################################

plot_calibration_instability <- function(cal_df, model_name) {
  ggplot2::ggplot() +
    ggplot2::geom_line(
      data = cal_df |>
        dplyr::filter(Type == "Bootstrap"),
      ggplot2::aes(
        x = mean_pred,
        y = obs,
        group = bootstrap
      ),
      alpha = 0.12,
      color = "grey50",
      linewidth = 0.7
    ) +
    ggplot2::geom_point(
      data = cal_df |>
        dplyr::filter(Type == "Apparent"),
      ggplot2::aes(
        x = mean_pred,
        y = obs
      ),
      size = 2
    ) +
    ggplot2::geom_line(
      data = cal_df |>
        dplyr::filter(Type == "Apparent"),
      ggplot2::aes(
        x = mean_pred,
        y = obs
      ),
      linewidth = 0.9
    ) +
    ggplot2::geom_abline(
      intercept = 0,
      slope = 1,
      linetype = 2
    ) +
    ggplot2::coord_equal() +
    ggplot2::scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, by = 0.2)
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, by = 0.2)
    ) +
    ggplot2::theme_bw() +
    ggplot2::labs(
      title = model_name,
      x = "Predicted risk",
      y = "Observed event rate"
    )
}

p_cal_logit <-
  plot_calibration_instability(
    all_results$Logistic$calibration_instability,
    "Logistic"
  )

p_cal_rf <-
  plot_calibration_instability(
    all_results$RandomForest$calibration_instability,
    "Random Forest"
  )

p_cal_xgb <-
  plot_calibration_instability(
    all_results$XGBoost$calibration_instability,
    "XGBoost"
  )

p_cal_all <-
  ggpubr::ggarrange(
    p_cal_logit,
    p_cal_rf,
    p_cal_xgb,
    ncol = 3
  )

print(p_cal_all)

################################################################################
## 24) INDIVIDUAL PREDICTION UNCERTAINTY VS PREDICTED RISK
################################################################################

p_uncertainty <-
  ggplot2::ggplot(
    patient_uncertainty_all,
    ggplot2::aes(
      x = pred_apparent,
      y = pred_boot_sd
    )
  ) +
  ggplot2::geom_point(alpha = 0.5) +
  ggplot2::geom_smooth(
    method = "loess",
    se = FALSE,
    linewidth = 0.9
  ) +
  ggplot2::facet_wrap(~ model) +
  ggplot2::theme_minimal() +
  ggplot2::labs(
    x = "Predicted risk",
    y = "Bootstrap SD",
    title = ""
  )

print(p_uncertainty)

################################################################################
## 25) DECISION FRAGILITY PLOT
################################################################################

p_fragility <-
  ggplot2::ggplot(
    fragility_all,
    ggplot2::aes(
      x = threshold,
      y = prop_fragile,
      color = Model,
      group = Model
    )
  ) +
  ggplot2::geom_line(linewidth = 1) +
  ggplot2::geom_point(size = 2.5) +
  ggplot2::theme_minimal() +
  ggplot2::labs(
    x = "Threshold probability",
    y = "Proportion with fragile decision",
    color = ""
  )

print(p_fragility)

################################################################################
## 26) DECISION CURVE ANALYSIS PLOT
################################################################################

dca_long <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(model_name) {
        all_results[[model_name]]$dca |>
          dplyr::transmute(
            Model = model_name,
            threshold,
            `Model apparent` = NB_apparent,
            `Model optimism-corrected` = NB_corrected,
            `Treat all` = NB_all,
            `Treat none` = NB_none
          )
      }
    )
  ) |>
  tidyr::pivot_longer(
    cols = c(
      `Model apparent`,
      `Model optimism-corrected`,
      `Treat all`,
      `Treat none`
    ),
    names_to = "Strategy",
    values_to = "NB"
  )

p_dca <-
  ggplot2::ggplot(
    dca_long,
    ggplot2::aes(
      x = threshold,
      y = NB,
      color = Strategy
    )
  ) +
  ggplot2::geom_line(linewidth = 0.9) +
  ggplot2::facet_wrap(~ Model) +
  ggplot2::coord_cartesian(
    ylim = c(-0.20, 0.10)
  ) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    legend.position = "bottom"
  ) +
  ggplot2::labs(
    x = "Threshold probability",
    y = "Net benefit",
    color = ""
  )

print(p_dca)

################################################################################
## 27) VARIABLE IMPORTANCE PLOTS
################################################################################

plot_importance_auc <- function(
    importance,
    model_name,
    top_n = 15) {

  plot_data <-
    importance |>
    dplyr::slice_head(n = top_n) |>
    dplyr::mutate(
      Feature = factor(
        Feature,
        levels = rev(Feature)
      )
    )

  ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
      x = Feature,
      y = mean_decrease_auc
    )
  ) +
    ggplot2::geom_col() +
    ggplot2::coord_flip() +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = model_name,
      x = "",
      y = "Mean decrease in AUC"
    )
}

p_imp_logit <-
  plot_importance_auc(
    all_results$Logistic$importance,
    "Logistic"
  )

p_imp_rf <-
  plot_importance_auc(
    all_results$RandomForest$importance,
    "Random Forest"
  )

p_imp_xgb <-
  plot_importance_auc(
    all_results$XGBoost$importance,
    "XGBoost"
  )

p_imp_all <-
  ggpubr::ggarrange(
    p_imp_logit,
    p_imp_rf,
    p_imp_xgb,
    ncol = 3
  )

print(p_imp_all)

################################################################################
## 28) COMBINE BOOTSTRAP DETAILS
################################################################################

performance_optimism_all <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(model_name) {
        all_results[[model_name]]$performance_optimism |>
          dplyr::mutate(
            Model = model_name,
            .before = 1
          )
      }
    )
  )

performance_corrected_dist_all <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(model_name) {
        all_results[[model_name]]$performance_corrected_dist |>
          dplyr::mutate(
            Model = model_name,
            .before = 1
          )
      }
    )
  )

target_sens_boot_all <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(model_name) {
        all_results[[model_name]]$target_sens_boot |>
          dplyr::mutate(
            Model = model_name,
            .before = 1
          )
      }
    )
  )

target_sens_optimism_all <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(model_name) {
        all_results[[model_name]]$target_sens_optimism |>
          dplyr::mutate(
            Model = model_name,
            .before = 1
          )
      }
    )
  )

target_sens_corrected_dist_all <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(model_name) {
        all_results[[model_name]]$target_sens_corrected_dist |>
          dplyr::mutate(
            Model = model_name,
            .before = 1
          )
      }
    )
  )

fail_log_all <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(model_name) {
        all_results[[model_name]]$fail_log |>
          dplyr::mutate(
            Model = model_name,
            .before = 1
          )
      }
    )
  )

################################################################################
## 29) EXPORT RESULTS
################################################################################

writexl::write_xlsx(
  list(
    Performance = performance_summary,
    Target_sensitivity = target_sens_summary,
    Uncertainty = uncertainty_summary,
    Patient_uncertainty = patient_uncertainty_all,
    Fragility = fragility_all,
    Net_benefit = dca_nb_table,
    NB_and_fragility = final_table,
    NB_fragility_wide = final_wide,
    Performance_optimism = performance_optimism_all,
    Corrected_performance_dist = performance_corrected_dist_all,
    Target_sens_boot = target_sens_boot_all,
    Target_sens_optimism = target_sens_optimism_all,
    Target_sens_corrected_dist = target_sens_corrected_dist_all,
    Variable_importance = dplyr::bind_rows(
      lapply(
        all_results,
        `[[`,
        "importance"
      )
    ),
    Fail_log = fail_log_all
  ),
  path = "CLADE_TRUFFLE_AUC_common_bootstrap_final.xlsx"
)

################################################################################
## 30) SAVE FIGURES
################################################################################

ggplot2::ggsave(
  "AUC_Calibration_instability.png",
  p_cal_all,
  width = 12,
  height = 4,
  dpi = 600
)

ggplot2::ggsave(
  "AUC_Individual_prediction_uncertainty.png",
  p_uncertainty,
  width = 10,
  height = 5,
  dpi = 600
)

ggplot2::ggsave(
  "AUC_Decision_fragility.png",
  p_fragility,
  width = 8,
  height = 5,
  dpi = 600
)

ggplot2::ggsave(
  "AUC_Decision_curve_analysis.png",
  p_dca,
  width = 10,
  height = 5,
  dpi = 600
)

ggplot2::ggsave(
  "AUC_Variable_importance.png",
  p_imp_all,
  width = 12,
  height = 4,
  dpi = 600
)

################################################################################
## END
################################################################################
