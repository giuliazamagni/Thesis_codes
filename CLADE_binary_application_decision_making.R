################################################################################
## CLADE MAIN ANALYSIS — PROBABILITY / CALIBRATION ORIENTED
## Elastic-net logistic regression vs Random Forest vs XGBoost
## Binary outcome: adverse pregnancy outcome
## Shared ordinary bootstrap for performance, uncertainty, fragility, and DCA
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

df$Caucasian    <- factor(df$Caucasian)
df$Diabetes_cat <- factor(df$Diabetes_cat)
df$Smoking      <- factor(df$Smoking)

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

## Number of bootstrap resamples
## Use 1000 for final analyses
B <- 1000

## Full DCA range
thresholds_dca <- seq(
  0.01,
  0.99,
  by = 0.01
)

## Clinically relevant thresholds for reporting
thresholds_interest <- c(
  0.05,
  0.08,
  0.10,
  0.15
)

## Permutations for variable importance
n_perm_importance <- 5


################################################################################
## 3) HELPER FUNCTIONS
################################################################################

clamp01 <- function(p) {
  
  pmin(
    pmax(p, 1e-15),
    1 - 1e-15
  )
}


safe_sd <- function(x) {
  
  if (sum(!is.na(x)) <= 1) {
    return(NA_real_)
  }
  
  stats::sd(
    x,
    na.rm = TRUE
  )
}


safe_q <- function(x, prob) {
  
  if (all(is.na(x))) {
    return(NA_real_)
  }
  
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
  
  if (length(unique(y)) < 2) {
    return(NA_real_)
  }
  
  as.numeric(
    pROC::roc(
      response = y,
      predictor = p,
      quiet = TRUE
    )$auc
  )
}


Brier_fun <- function(y, p) {
  
  mean(
    (p - y)^2,
    na.rm = TRUE
  )
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
  
  if (length(unique(y)) < 2) {
    return(NA_real_)
  }
  
  if (any(!is.finite(lp))) {
    return(NA_real_)
  }
  
  fit <- try(
    stats::glm(
      y ~ lp,
      family = stats::binomial()
    ),
    silent = TRUE
  )
  
  if (inherits(fit, "try-error")) {
    return(NA_real_)
  }
  
  co <- try(
    stats::coef(fit),
    silent = TRUE
  )
  
  if (inherits(co, "try-error")) {
    return(NA_real_)
  }
  
  if (
    length(co) < 2 ||
    !is.finite(co[2])
  ) {
    return(NA_real_)
  }
  
  unname(co[2])
}


CalIntercept <- function(y, p) {
  
  p <- clamp01(p)
  
  lp <- qlogis(p)
  
  if (length(unique(y)) < 2) {
    return(NA_real_)
  }
  
  if (any(!is.finite(lp))) {
    return(NA_real_)
  }
  
  fit <- try(
    stats::glm(
      y ~ offset(lp),
      family = stats::binomial()
    ),
    silent = TRUE
  )
  
  if (inherits(fit, "try-error")) {
    return(NA_real_)
  }
  
  co <- try(
    stats::coef(fit),
    silent = TRUE
  )
  
  if (inherits(co, "try-error")) {
    return(NA_real_)
  }
  
  if (
    length(co) < 1 ||
    !is.finite(co[1])
  ) {
    return(NA_real_)
  }
  
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
## 5) SHARED ORDINARY BOOTSTRAP
##
## Bootstrap subjects from the full sample.
## Event prevalence is allowed to vary naturally across bootstrap samples.
##
## Importantly, these exact same resamples will be used for all models.
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

## Check distribution of prevalence across bootstrap samples
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
## 6) MODEL MATRIX
################################################################################

make_mm <- function(newdata, template_cols) {
  
  mm <- stats::model.matrix(
    ~ . - 1,
    data = newdata
  )
  
  missing_cols <- setdiff(
    template_cols,
    colnames(mm)
  )
  
  if (length(missing_cols) > 0) {
    
    mm_missing <- matrix(
      0,
      nrow = nrow(mm),
      ncol = length(missing_cols)
    )
    
    colnames(mm_missing) <- missing_cols
    
    mm <- cbind(
      mm,
      mm_missing
    )
  }
  
  mm <- mm[
    ,
    template_cols,
    drop = FALSE
  ]
  
  mm
}


X_full <- stats::model.matrix(
  ~ . - 1,
  X
)

mm_cols <- colnames(X_full)


################################################################################
## 7) MODEL FIT AND PREDICTION FUNCTIONS
################################################################################


############################
## 7A) ELASTIC-NET LOGISTIC
############################

fit_logit <- function(idx, seed = NULL) {
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  x_boot <- X_full[
    idx,
    ,
    drop = FALSE
  ]
  
  y_boot <- y[idx]
  
  ## Need both outcome classes
  if (length(unique(y_boot)) < 2) {
    stop("Only one outcome class present.")
  }
  
  cv_logit <- glmnet::cv.glmnet(
    x = x_boot,
    y = y_boot,
    family = "binomial",
    alpha = 0.5
  )
  
  glmnet::glmnet(
    x = x_boot,
    y = y_boot,
    family = "binomial",
    alpha = 0.5,
    lambda = cv_logit$lambda.min
  )
}


pred_logit <- function(model, newdata) {
  
  mm_new <- make_mm(
    newdata,
    mm_cols
  )
  
  as.numeric(
    stats::predict(
      model,
      newx = mm_new,
      type = "response"
    )
  )
}


############################
## 7B) RANDOM FOREST
############################

fit_rf <- function(idx, seed = NULL) {
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  df_boot <- df[
    idx,
    ,
    drop = FALSE
  ]
  
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
    num.trees = 600,
    mtry = 6,
    min.node.size = 40,
    max.depth = 4,
    sample.fraction = 0.8,
    splitrule = "gini",
    seed = seed
  )
}


pred_rf <- function(model, newdata) {
  
  pr <- stats::predict(
    model,
    data = newdata
  )$predictions
  
  if ("1" %in% colnames(pr)) {
    
    return(
      as.numeric(
        pr[, "1"]
      )
    )
  }
  
  as.numeric(
    pr[, ncol(pr)]
  )
}


############################
## 7C) XGBOOST
############################

fit_xgb <- function(idx, seed = NULL) {
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  x_boot <- X_full[
    idx,
    ,
    drop = FALSE
  ]
  
  y_boot <- y[idx]
  
  if (length(unique(y_boot)) < 2) {
    stop("Only one outcome class present.")
  }
  
  dtrain <- xgboost::xgb.DMatrix(
    data = x_boot,
    label = y_boot
  )
  
  cv <- xgboost::xgb.cv(
    data = dtrain,
    objective = "binary:logistic",
    eval_metric = "logloss",
    nrounds = 1200,
    eta = 0.03,
    max_depth = 2,
    min_child_weight = 50,
    subsample = 0.9,
    colsample_bytree = 0.8,
    lambda = 5,
    alpha = 1,
    nfold = 5,
    early_stopping_rounds = 50,
    verbose = 0
  )
  
  best_nrounds <- cv$best_iteration
  
  if (
    is.null(best_nrounds) ||
    is.na(best_nrounds) ||
    best_nrounds < 1
  ) {
    best_nrounds <- 50
  }
  
  xgboost::xgb.train(
    data = dtrain,
    nrounds = best_nrounds,
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = 0.03,
    max_depth = 2,
    min_child_weight = 50,
    subsample = 0.9,
    colsample_bytree = 0.8,
    lambda = 5,
    alpha = 1,
    verbose = 0
  )
}


pred_xgb <- function(model, newdata) {
  
  mm_new <- make_mm(
    newdata,
    mm_cols
  )
  
  as.numeric(
    stats::predict(
      model,
      xgboost::xgb.DMatrix(mm_new)
    )
  )
}


################################################################################
## 8) MODEL LIST
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
## 9) DECISION CURVE ANALYSIS
################################################################################

decision_curve_data <- function(
    y,
    p,
    thresholds = thresholds_dca) {
  
  n_local <- length(y)
  
  prevalence <- mean(
    y,
    na.rm = TRUE
  )
  
  tibble::tibble(
    threshold = thresholds
  ) |>
    
    dplyr::mutate(
      
      TP = purrr::map_dbl(
        threshold,
        ~ sum(
          p >= .x &
            y == 1,
          na.rm = TRUE
        )
      ),
      
      FP = purrr::map_dbl(
        threshold,
        ~ sum(
          p >= .x &
            y == 0,
          na.rm = TRUE
        )
      ),
      
      NB_model =
        (TP / n_local) -
        (FP / n_local) *
        threshold /
        (1 - threshold),
      
      NB_all =
        prevalence -
        (1 - prevalence) *
        threshold /
        (1 - threshold),
      
      NB_none = 0
    )
}


################################################################################
## 10) PATIENT-LEVEL PREDICTION UNCERTAINTY
################################################################################

summarise_patient_bootstrap <- function(
    pred_boot,
    apparent_pred,
    y,
    model_name) {
  
  tibble::tibble(
    
    patient_id =
      seq_len(
        nrow(pred_boot)
      ),
    
    outcome =
      y,
    
    pred_apparent =
      apparent_pred,
    
    pred_boot_mean =
      rowMeans(
        pred_boot,
        na.rm = TRUE
      ),
    
    pred_boot_sd =
      apply(
        pred_boot,
        1,
        safe_sd
      ),
    
    pred_boot_p025 =
      apply(
        pred_boot,
        1,
        safe_q,
        prob = 0.025
      ),
    
    pred_boot_p975 =
      apply(
        pred_boot,
        1,
        safe_q,
        prob = 0.975
      )
    
  ) |>
    
    dplyr::mutate(
      
      pred_boot_interval_width =
        pred_boot_p975 -
        pred_boot_p025,
      
      model =
        model_name
    )
}


################################################################################
## 11) DECISION FRAGILITY
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
        
        Model =
          model_name,
        
        threshold =
          thr,
        
        n_fragile =
          sum(
            fragile,
            na.rm = TRUE
          ),
        
        n_total =
          sum(
            !is.na(fragile)
          ),
        
        prop_fragile =
          mean(
            fragile,
            na.rm = TRUE
          ),
        
        n_above =
          sum(
            patient_tbl$pred_boot_p025 > thr,
            na.rm = TRUE
          ),
        
        n_below =
          sum(
            patient_tbl$pred_boot_p975 < thr,
            na.rm = TRUE
          )
      )
    }
  )
}


################################################################################
## 12) CALIBRATION INSTABILITY
################################################################################

make_breaks_deciles <- function(
    p,
    g = 10) {
  
  qs <- unique(
    stats::quantile(
      p,
      probs = seq(
        0,
        1,
        length.out = g + 1
      ),
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


assign_groups <- function(
    p,
    breaks) {
  
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
  
  breaks <- make_breaks_deciles(
    apparent_pred,
    g
  )
  
  apparent_df <-
    tibble::tibble(
      y = y,
      p = apparent_pred
    ) |>
    
    dplyr::mutate(
      group =
        assign_groups(
          p,
          breaks
        )
    ) |>
    
    dplyr::group_by(
      group
    ) |>
    
    dplyr::summarise(
      mean_pred =
        mean(
          p,
          na.rm = TRUE
        ),
      obs =
        mean(
          y,
          na.rm = TRUE
        ),
      .groups = "drop"
    ) |>
    
    dplyr::mutate(
      Type = "Apparent",
      bootstrap = NA_integer_
    )
  
  
  boot_df <- purrr::map_dfr(
    seq_len(
      ncol(pred_boot)
    ),
    function(j) {
      
      tibble::tibble(
        y = y,
        p = pred_boot[, j]
      ) |>
        
        dplyr::mutate(
          group =
            assign_groups(
              p,
              breaks
            )
        ) |>
        
        dplyr::group_by(
          group
        ) |>
        
        dplyr::summarise(
          mean_pred =
            mean(
              p,
              na.rm = TRUE
            ),
          obs =
            mean(
              y,
              na.rm = TRUE
            ),
          .groups = "drop"
        ) |>
        
        dplyr::mutate(
          Type = "Bootstrap",
          bootstrap = j
        )
    }
  )
  
  dplyr::bind_rows(
    apparent_df,
    boot_df
  )
}


################################################################################
## 13) LOG-LOSS PERMUTATION IMPORTANCE
################################################################################

permutation_importance_logloss <- function(
    model,
    pred_fun,
    X,
    y,
    model_name,
    n_perm = 5) {
  
  cat(
    "Permutation importance:",
    model_name,
    "\n"
  )
  
  p_ref <- clamp01(
    pred_fun(
      model,
      X
    )
  )
  
  base_logloss <-
    LogLoss_fun(
      y,
      p_ref
    )
  
  vars <- colnames(X)
  
  pb <- make_progress(
    length(vars)
  )
  
  on.exit(
    try(
      close(pb),
      silent = TRUE
    ),
    add = TRUE
  )
  
  out <- purrr::map_dfr(
    seq_along(vars),
    function(j) {
      
      utils::setTxtProgressBar(
        pb,
        j
      )
      
      variable <- vars[j]
      
      increases <- rep(
        NA_real_,
        n_perm
      )
      
      for (k in seq_len(n_perm)) {
        
        X_perm <- X
        
        X_perm[[variable]] <-
          sample(
            X_perm[[variable]],
            replace = FALSE
          )
        
        p_perm <- try(
          clamp01(
            pred_fun(
              model,
              X_perm
            )
          ),
          silent = TRUE
        )
        
        if (inherits(p_perm, "try-error")) {
          next
        }
        
        increases[k] <-
          LogLoss_fun(
            y,
            p_perm
          ) -
          base_logloss
      }
      
      tibble::tibble(
        
        Model =
          model_name,
        
        Feature =
          variable,
        
        base_logloss =
          base_logloss,
        
        mean_increase_logloss =
          mean(
            increases,
            na.rm = TRUE
          ),
        
        sd_increase_logloss =
          stats::sd(
            increases,
            na.rm = TRUE
          ),
        
        q025_increase_logloss =
          safe_q(
            increases,
            0.025
          ),
        
        q975_increase_logloss =
          safe_q(
            increases,
            0.975
          ),
        
        n_valid_perm =
          sum(
            !is.na(increases)
          )
      )
    }
  )
  
  cat("\n")
  
  out |>
    dplyr::arrange(
      dplyr::desc(
        mean_increase_logloss
      )
    )
}


################################################################################
## 14) SHARED-BOOTSTRAP PIPELINE FOR ONE MODEL
##
## EACH BOOTSTRAP MODEL IS FIT ONLY ONCE.
##
## That single fit is then used for:
##
##   - bootstrap-sample performance
##   - original-sample performance
##   - patient-level prediction uncertainty
##   - DCA in bootstrap sample
##   - DCA in original sample
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
    n_perm_importance = 5,
    base_seed = 10000) {
  
  cat("\n")
  cat("====================================================\n")
  cat("MODEL:", model_name, "\n")
  cat("====================================================\n")
  
  
  ##########################################################################
  ## 14A) APPARENT MODEL
  ##########################################################################
  
  apparent_model <- fit_fun(
    seq_len(nrow(X)),
    seed = base_seed
  )
  
  apparent_pred <- clamp01(
    pred_fun(
      apparent_model,
      X
    )
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
  
  
  ##########################################################################
  ## 14B) STORAGE
  ##########################################################################
  
  B_local <- length(
    bootstrap_indices
  )
  
  pred_boot <- matrix(
    NA_real_,
    nrow = nrow(X),
    ncol = B_local
  )
  
  performance_list <- vector(
    "list",
    B_local
  )
  
  dca_list <- vector(
    "list",
    B_local
  )
  
  valid <- logical(
    B_local
  )
  
  fail_log <- tibble::tibble(
    bootstrap = integer(),
    stage = character(),
    message = character()
  )
  
  
  ##########################################################################
  ## 14C) ONE BOOTSTRAP LOOP
  ##########################################################################
  
  pb <- make_progress(
    B_local
  )
  
  on.exit(
    try(
      close(pb),
      silent = TRUE
    ),
    add = TRUE
  )
  
  
  for (b in seq_len(B_local)) {
    
    utils::setTxtProgressBar(
      pb,
      b
    )
    
    idxb <-
      bootstrap_indices[[b]]
    
    
    ########################################################################
    ## Fit model ONCE
    ########################################################################
    
    model_b <- try(
      fit_fun(
        idxb,
        seed = base_seed + b
      ),
      silent = TRUE
    )
    
    if (inherits(model_b, "try-error")) {
      
      fail_log <- bind_rows(
        fail_log,
        tibble(
          bootstrap = b,
          stage = "fit",
          message = as.character(model_b)
        )
      )
      
      next
    }
    
    
    ########################################################################
    ## Prediction on original dataset
    ##
    ## Used simultaneously for:
    ##   - test performance
    ##   - patient uncertainty
    ##   - test DCA
    ########################################################################
    
    p_original <- try(
      clamp01(
        pred_fun(
          model_b,
          X
        )
      ),
      silent = TRUE
    )
    
    if (
      inherits(p_original, "try-error") ||
      any(!is.finite(p_original))
    ) {
      
      fail_log <- bind_rows(
        fail_log,
        tibble(
          bootstrap = b,
          stage = "predict_original",
          message = "Prediction failed or returned non-finite values."
        )
      )
      
      next
    }
    
    
    ########################################################################
    ## Prediction within bootstrap sample
    ##
    ## Same fitted model, evaluated on its own development sample.
    ########################################################################
    
    p_bootsample <-
      p_original[idxb]
    
    
    ########################################################################
    ## Store patient-level predictions
    ########################################################################
    
    pred_boot[, b] <-
      p_original
    
    
    ########################################################################
    ## Performance
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
      bind_cols(
        
        tibble(
          bootstrap = b,
          bootstrap_prevalence =
            mean(
              y[idxb]
            )
        ),
        
        perf_boot |>
          rename_with(
            ~ paste0(
              .x,
              "_boot"
            )
          ),
        
        perf_test |>
          rename_with(
            ~ paste0(
              .x,
              "_test"
            )
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
      
      select(
        threshold,
        NB_boot = NB_model
      )
    
    
    dca_test <-
      decision_curve_data(
        y = y,
        p = p_original,
        thresholds = thresholds_dca
      ) |>
      
      select(
        threshold,
        NB_test = NB_model
      )
    
    
    dca_list[[b]] <-
      dca_boot |>
      
      left_join(
        dca_test,
        by = "threshold"
      ) |>
      
      mutate(
        bootstrap = b,
        NB_optimism =
          NB_boot -
          NB_test,
        .before = 1
      )
    
    
    valid[b] <- TRUE
  }
  
  cat("\n")
  
  
  ##########################################################################
  ## 14D) KEEP VALID REPLICATIONS
  ##########################################################################
  
  if (sum(valid) < 2) {
    
    stop(
      paste(
        "Too few valid bootstrap replications for",
        model_name
      )
    )
  }
  
  pred_boot <-
    pred_boot[
      ,
      valid,
      drop = FALSE
    ]
  
  valid_bootstrap_ids <-
    which(valid)
  
  colnames(pred_boot) <-
    paste0(
      "boot_",
      valid_bootstrap_ids
    )
  
  performance_boot <-
    bind_rows(
      performance_list
    )
  
  dca_boot_detail <-
    bind_rows(
      dca_list
    )
  
  cat(
    "Valid bootstrap replications:",
    sum(valid),
    "/",
    B_local,
    "\n"
  )
  
  
  ##########################################################################
  ## 14E) OPTIMISM-CORRECTED PERFORMANCE
  ##########################################################################
  
  performance_optimism <-
    performance_boot |>
    
    transmute(
      
      bootstrap,
      
      AUC_optimism =
        AUC_boot -
        AUC_test,
      
      Brier_optimism =
        Brier_boot -
        Brier_test,
      
      CalIntercept_optimism =
        CalIntercept_boot -
        CalIntercept_test,
      
      CalSlope_optimism =
        CalSlope_boot -
        CalSlope_test
    )
  
  
  performance_corrected_dist <-
    performance_optimism |>
    
    transmute(
      
      bootstrap,
      
      AUC_corrected_iter =
        apparent_metrics$AUC -
        AUC_optimism,
      
      Brier_corrected_iter =
        apparent_metrics$Brier -
        Brier_optimism,
      
      CalIntercept_corrected_iter =
        apparent_metrics$CalIntercept -
        CalIntercept_optimism,
      
      CalSlope_corrected_iter =
        apparent_metrics$CalSlope -
        CalSlope_optimism
    )
  
  
  performance_summary <-
    tibble(
      
      Model =
        model_name,
      
      AUC_apparent =
        apparent_metrics$AUC,
      
      AUC_corrected =
        apparent_metrics$AUC -
        mean(
          performance_optimism$AUC_optimism,
          na.rm = TRUE
        ),
      
      AUC_p025 =
        safe_q(
          performance_corrected_dist$AUC_corrected_iter,
          0.025
        ),
      
      AUC_p975 =
        safe_q(
          performance_corrected_dist$AUC_corrected_iter,
          0.975
        ),
      
      
      Brier_apparent =
        apparent_metrics$Brier,
      
      Brier_corrected =
        apparent_metrics$Brier -
        mean(
          performance_optimism$Brier_optimism,
          na.rm = TRUE
        ),
      
      Brier_p025 =
        safe_q(
          performance_corrected_dist$Brier_corrected_iter,
          0.025
        ),
      
      Brier_p975 =
        safe_q(
          performance_corrected_dist$Brier_corrected_iter,
          0.975
        ),
      
      
      CalIntercept_apparent =
        apparent_metrics$CalIntercept,
      
      CalIntercept_corrected =
        apparent_metrics$CalIntercept -
        mean(
          performance_optimism$CalIntercept_optimism,
          na.rm = TRUE
        ),
      
      CalIntercept_p025 =
        safe_q(
          performance_corrected_dist$CalIntercept_corrected_iter,
          0.025
        ),
      
      CalIntercept_p975 =
        safe_q(
          performance_corrected_dist$CalIntercept_corrected_iter,
          0.975
        ),
      
      
      CalSlope_apparent =
        apparent_metrics$CalSlope,
      
      CalSlope_corrected =
        apparent_metrics$CalSlope -
        mean(
          performance_optimism$CalSlope_optimism,
          na.rm = TRUE
        ),
      
      CalSlope_p025 =
        safe_q(
          performance_corrected_dist$CalSlope_corrected_iter,
          0.025
        ),
      
      CalSlope_p975 =
        safe_q(
          performance_corrected_dist$CalSlope_corrected_iter,
          0.975
        ),
      
      n_valid_bootstrap =
        sum(valid)
    )
  
  
  ##########################################################################
  ## 14F) PATIENT-LEVEL UNCERTAINTY
  ##########################################################################
  
  patient_uncertainty <-
    summarise_patient_bootstrap(
      
      pred_boot =
        pred_boot,
      
      apparent_pred =
        apparent_pred,
      
      y =
        y,
      
      model_name =
        model_name
    )
  
  
  ##########################################################################
  ## 14G) DECISION FRAGILITY
  ##########################################################################
  
  fragility <-
    compute_fragility(
      
      patient_tbl =
        patient_uncertainty,
      
      thresholds =
        thresholds_interest,
      
      model_name =
        model_name
    )
  
  
  ##########################################################################
  ## 14H) CALIBRATION INSTABILITY
  ##########################################################################
  
  calibration_instability <-
    calibration_instability_data(
      
      apparent_pred =
        apparent_pred,
      
      pred_boot =
        pred_boot,
      
      y =
        y,
      
      g =
        10
    )
  
  
  ##########################################################################
  ## 14I) OPTIMISM-CORRECTED DCA
  ##########################################################################
  
  dca_optimism <-
    dca_boot_detail |>
    
    group_by(
      threshold
    ) |>
    
    summarise(
      
      mean_NB_optimism =
        mean(
          NB_optimism,
          na.rm = TRUE
        ),
      
      n_valid =
        sum(
          !is.na(
            NB_optimism
          )
        ),
      
      .groups =
        "drop"
    )
  
  
  dca_corrected <-
    apparent_dca |>
    
    left_join(
      dca_optimism,
      by = "threshold"
    ) |>
    
    mutate(
      
      NB_corrected =
        NB_apparent -
        mean_NB_optimism
    )
  
  
  ##########################################################################
  ## 14J) VARIABLE IMPORTANCE
  ##########################################################################
  
  importance <-
    permutation_importance_logloss(
      
      model =
        apparent_model,
      
      pred_fun =
        pred_fun,
      
      X =
        X,
      
      y =
        y,
      
      model_name =
        model_name,
      
      n_perm =
        n_perm_importance
    )
  
  
  ##########################################################################
  ## RETURN
  ##########################################################################
  
  list(
    
    apparent_model =
      apparent_model,
    
    apparent_pred =
      apparent_pred,
    
    performance_summary =
      performance_summary,
    
    performance_boot =
      performance_boot,
    
    performance_optimism =
      performance_optimism,
    
    performance_corrected_dist =
      performance_corrected_dist,
    
    pred_boot =
      pred_boot,
    
    patient_uncertainty =
      patient_uncertainty,
    
    fragility =
      fragility,
    
    calibration_instability =
      calibration_instability,
    
    dca =
      dca_corrected,
    
    dca_boot_detail =
      dca_boot_detail,
    
    importance =
      importance,
    
    fail_log =
      fail_log,
    
    valid_bootstrap_ids =
      valid_bootstrap_ids
  )
}


################################################################################
## 15) RUN ALL MODELS
################################################################################

all_results <- list()

for (model_name in names(models)) {
  
  all_results[[model_name]] <-
    run_model_pipeline(
      
      model_name =
        model_name,
      
      fit_fun =
        models[[model_name]]$fit,
      
      pred_fun =
        models[[model_name]]$pred,
      
      X =
        X,
      
      y =
        y,
      
      bootstrap_indices =
        bootstrap_indices,
      
      thresholds_dca =
        thresholds_dca,
      
      thresholds_interest =
        thresholds_interest,
      
      n_perm_importance =
        n_perm_importance,
      
      base_seed =
        10000
    )
}


################################################################################
## 16) COMBINE PERFORMANCE RESULTS
################################################################################

performance_summary <-
  bind_rows(
    
    lapply(
      all_results,
      `[[`,
      "performance_summary"
    )
  )

print(
  performance_summary
)


################################################################################
## 17) COMBINE PATIENT-LEVEL UNCERTAINTY
################################################################################

patient_uncertainty_all <-
  bind_rows(
    
    lapply(
      all_results,
      `[[`,
      "patient_uncertainty"
    )
  )


uncertainty_summary <-
  patient_uncertainty_all |>
  
  group_by(
    model
  ) |>
  
  summarise(
    
    n =
      n(),
    
    mean_pred_sd =
      mean(
        pred_boot_sd,
        na.rm = TRUE
      ),
    
    median_pred_sd =
      median(
        pred_boot_sd,
        na.rm = TRUE
      ),
    
    q90_pred_sd =
      as.numeric(
        quantile(
          pred_boot_sd,
          0.90,
          na.rm = TRUE
        )
      ),
    
    q95_pred_sd =
      as.numeric(
        quantile(
          pred_boot_sd,
          0.95,
          na.rm = TRUE
        )
      ),
    
    mean_interval_width =
      mean(
        pred_boot_interval_width,
        na.rm = TRUE
      ),
    
    median_interval_width =
      median(
        pred_boot_interval_width,
        na.rm = TRUE
      ),
    
    .groups =
      "drop"
  ) |>
  
  rename(
    Model = model
  )

print(
  uncertainty_summary
)


################################################################################
## 18) COMBINE FRAGILITY RESULTS
################################################################################

fragility_all <-
  bind_rows(
    
    lapply(
      all_results,
      `[[`,
      "fragility"
    )
  )

fragility_all <-
  fragility_all |>
  
  mutate(
    Fragility_percent =
      100 *
      prop_fragile
  )

print(
  fragility_all
)


################################################################################
## 19) NET BENEFIT AT PRESPECIFIED THRESHOLDS
################################################################################

dca_nb_table <-
  bind_rows(
    
    lapply(
      names(all_results),
      function(model_name) {
        
        all_results[[model_name]]$dca |>
          
          mutate(
            threshold_round =
              round(
                threshold,
                2
              )
          ) |>
          
          filter(
            threshold_round %in%
              thresholds_interest
          ) |>
          
          transmute(
            Model =
              model_name,
            
            threshold =
              threshold_round,
            
            NB_apparent =
              NB_apparent,
            
            NB_corrected =
              NB_corrected,
            
            NB_all =
              NB_all,
            
            NB_none =
              NB_none
          )
      }
    )
  )

print(
  dca_nb_table
)


################################################################################
## 20) FINAL NET BENEFIT AND FRAGILITY TABLE
################################################################################

final_table <-
  dca_nb_table |>
  
  left_join(
    
    fragility_all |>
      
      select(
        Model,
        threshold,
        n_fragile,
        Fragility_percent
      ),
    
    by =
      c(
        "Model",
        "threshold"
      )
  ) |>
  
  arrange(
    Model,
    threshold
  )

print(
  final_table
)


final_wide <-
  final_table |>
  
  select(
    Model,
    threshold,
    NB_corrected,
    Fragility_percent
  ) |>
  
  pivot_wider(
    
    names_from =
      threshold,
    
    values_from =
      c(
        NB_corrected,
        Fragility_percent
      ),
    
    names_glue =
      "t={threshold}_{.value}"
  )

print(
  final_wide
)


################################################################################
## 21) CALIBRATION INSTABILITY PLOTS
################################################################################

plot_calibration_instability <- function(
    cal_df,
    model_name) {
  
  ggplot() +
    
    geom_line(
      
      data =
        cal_df |>
        filter(
          Type == "Bootstrap"
        ),
      
      aes(
        x = mean_pred,
        y = obs,
        group = bootstrap
      ),
      
      alpha = 0.12,
      color = "grey50",
      linewidth = 0.7
    ) +
    
    geom_point(
      
      data =
        cal_df |>
        filter(
          Type == "Apparent"
        ),
      
      aes(
        x = mean_pred,
        y = obs
      ),
      
      size = 2
    ) +
    
    geom_line(
      
      data =
        cal_df |>
        filter(
          Type == "Apparent"
        ),
      
      aes(
        x = mean_pred,
        y = obs
      ),
      
      linewidth = 0.9
    ) +
    
    geom_abline(
      intercept = 0,
      slope = 1,
      linetype = 2
    ) +
    
    coord_equal() +
    
    scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(
        0,
        1,
        by = 0.2
      )
    ) +
    
    scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(
        0,
        1,
        by = 0.2
      )
    ) +
    
    theme_bw() +
    
    labs(
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

print(
  p_cal_all
)


################################################################################
## 22) INDIVIDUAL UNCERTAINTY VS PREDICTED RISK
################################################################################

p_uncertainty <-
  ggplot(
    patient_uncertainty_all,
    aes(
      x = pred_apparent,
      y = pred_boot_sd
    )
  ) +
  
  geom_point(
    alpha = 0.5
  ) +
  
  geom_smooth(
    method = "loess",
    se = FALSE,
    linewidth = 0.9
  ) +
  
  facet_wrap(
    ~ model
  ) +
  
  theme_minimal() +
  
  labs(
    x = "Predicted risk",
    y = "Bootstrap SD",
    title = ""
  )

print(
  p_uncertainty
)


################################################################################
## 23) DECISION FRAGILITY PLOT
################################################################################

p_fragility <-
  ggplot(
    fragility_all,
    aes(
      x = threshold,
      y = prop_fragile,
      color = Model,
      group = Model
    )
  ) +
  
  geom_line(
    linewidth = 1
  ) +
  
  geom_point(
    size = 2.5
  ) +
  
  theme_minimal() +
  
  labs(
    x = "Threshold probability",
    y = "Proportion with fragile decision",
    color = ""
  )

print(
  p_fragility
)


################################################################################
## 24) DECISION CURVE PLOT
################################################################################

dca_long <-
  bind_rows(
    
    lapply(
      names(all_results),
      function(model_name) {
        
        all_results[[model_name]]$dca |>
          
          transmute(
            Model =
              model_name,
            
            threshold,
            
            `Model apparent` =
              NB_apparent,
            
            `Model optimism-corrected` =
              NB_corrected,
            
            `Treat all` =
              NB_all,
            
            `Treat none` =
              NB_none
          )
      }
    )
  ) |>
  
  pivot_longer(
    
    cols =
      c(
        `Model apparent`,
        `Model optimism-corrected`,
        `Treat all`,
        `Treat none`
      ),
    
    names_to =
      "Strategy",
    
    values_to =
      "NB"
  )


p_dca <-
  ggplot(
    dca_long,
    aes(
      x = threshold,
      y = NB,
      color = Strategy
    )
  ) +
  
  geom_line(
    linewidth = 0.9
  ) +
  
  facet_wrap(
    ~ Model
  ) +
  
  coord_cartesian(
    ylim = c(
      -0.02,
      0.10
    )
  ) +
  
  theme_minimal() +
  
  theme(
    legend.position = "bottom"
  ) +
  
  labs(
    x = "Threshold probability",
    y = "Net benefit",
    color = ""
  )

print(
  p_dca
)


################################################################################
## 25) VARIABLE IMPORTANCE PLOTS
################################################################################

plot_importance <- function(
    importance,
    model_name,
    top_n = 15) {
  
  plot_data <-
    importance |>
    
    slice_head(
      n = top_n
    ) |>
    
    mutate(
      Feature =
        factor(
          Feature,
          levels =
            rev(
              Feature
            )
        )
    )
  
  ggplot(
    plot_data,
    aes(
      x = Feature,
      y = mean_increase_logloss
    )
  ) +
    
    geom_col() +
    
    coord_flip() +
    
    theme_minimal() +
    
    labs(
      title = model_name,
      x = "",
      y = "Mean increase in logloss"
    )
}


p_imp_logit <-
  plot_importance(
    all_results$Logistic$importance,
    "Logistic"
  )

p_imp_rf <-
  plot_importance(
    all_results$RandomForest$importance,
    "Random Forest"
  )

p_imp_xgb <-
  plot_importance(
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

print(
  p_imp_all
)


################################################################################
## 26) COMBINE BOOTSTRAP DETAILS
################################################################################

performance_optimism_all <-
  bind_rows(
    
    lapply(
      names(all_results),
      function(model_name) {
        
        all_results[[model_name]]$performance_optimism |>
          
          mutate(
            Model = model_name,
            .before = 1
          )
      }
    )
  )


performance_corrected_dist_all <-
  bind_rows(
    
    lapply(
      names(all_results),
      function(model_name) {
        
        all_results[[model_name]]$performance_corrected_dist |>
          
          mutate(
            Model = model_name,
            .before = 1
          )
      }
    )
  )


fail_log_all <-
  bind_rows(
    
    lapply(
      names(all_results),
      function(model_name) {
        
        all_results[[model_name]]$fail_log |>
          
          mutate(
            Model = model_name,
            .before = 1
          )
      }
    )
  )


################################################################################
## 27) EXPORT RESULTS
################################################################################

writexl::write_xlsx(
  
  list(
    
    Performance =
      performance_summary,
    
    Uncertainty =
      uncertainty_summary,
    
    Patient_uncertainty =
      patient_uncertainty_all,
    
    Fragility =
      fragility_all,
    
    Net_benefit =
      dca_nb_table,
    
    NB_and_fragility =
      final_table,
    
    NB_fragility_wide =
      final_wide,
    
    Performance_optimism =
      performance_optimism_all,
    
    Corrected_performance_dist =
      performance_corrected_dist_all,
    
    Variable_importance =
      bind_rows(
        lapply(
          all_results,
          `[[`,
          "importance"
        )
      ),
    
    Fail_log =
      fail_log_all
  ),
  
  path =
    "CLADE_TRUFFLE_bootstrap_final.xlsx"
)


################################################################################
## 28) SAVE FIGURES
################################################################################

ggsave(
  "Calibration_instability.png",
  p_cal_all,
  width = 12,
  height = 4,
  dpi = 600
)

ggsave(
  "Individual_prediction_uncertainty.png",
  p_uncertainty,
  width = 10,
  height = 5,
  dpi = 600
)

ggsave(
  "Decision_fragility.png",
  p_fragility,
  width = 8,
  height = 5,
  dpi = 600
)

ggsave(
  "Decision_curve_analysis.png",
  p_dca,
  width = 10,
  height = 5,
  dpi = 600
)

ggsave(
  "Variable_importance.png",
  p_imp_all,
  width = 12,
  height = 4,
  dpi = 600
)


################################################################################
## END
################################################################################