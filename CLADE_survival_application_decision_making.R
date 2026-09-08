################################################################################
## CLADE SURVIVAL APPLICATION — PROBABILITY / CALIBRATION ORIENTED
## Cox proportional hazards vs Random Survival Forest
## Endpoint: AMI or stroke
## Prediction horizon: 36 months
################################################################################

library(haven)
library(dplyr)
library(tidyr)
library(purrr)
library(tibble)
library(survival)
library(ranger)
library(riskRegression)
library(ggplot2)
library(writexl)
library(ggpubr)

set.seed(123)

################################################################################
## 0) ANALYSIS SETTINGS
################################################################################

MODEL_ORIENTATION <- "probability_calibration_CoxPH_RSF_shared_ordinary_boot500"

B <- 500

n_perm_importance <- 5

rf_tune_trees  <- 750
rf_final_trees <- 1500

time_horizon <- 36


thresholds_dca <- seq(0.01, 0.99, by = 0.01)
thresholds_to_test <- c(0.05, 0.08, 0.10, 0.15, 0.20, 0.25)
thresholds_interest <- c(0.05, 0.08, 0.10, 0.15, 0.20, 0.25)

################################################################################
## 1) LOAD AND PREPROCESS DATA
################################################################################

df <- read_dta("INCLISAN_ready.dta") |>
  as.data.frame() |>
  dplyr::select(
    age_at_LDL_index,
    ANA_SESSO,
    PAS,
    colesterolo_mmol,
    HDL_mmol,
    diab_index,
    FR_FUMO,
    fup_D_IMA_STROKE,
    death_IMA_Stroke
  ) |>
  stats::na.omit() |>
  as.data.frame()

df$ANA_SESSO <- factor(df$ANA_SESSO)
df$diab_index <- factor(df$diab_index)
df$FR_FUMO <- factor(df$FR_FUMO)

stopifnot(all(df$death_IMA_Stroke %in% c(0, 1)))
stopifnot(all(df$fup_D_IMA_STROKE >= 0))

time_var <- "fup_D_IMA_STROKE"
event_var <- "death_IMA_Stroke"

predictor_vars <- c(
  "age_at_LDL_index",
  "ANA_SESSO",
  "PAS",
  "colesterolo_mmol",
  "HDL_mmol",
  "diab_index",
  "FR_FUMO"
)

cat("N =", nrow(df), "\n")
cat("Number of predictors =", length(predictor_vars), "\n")
cat("Total events =", sum(df[[event_var]] == 1), "\n")
cat("Crude overall event rate =", round(mean(df[[event_var]] == 1), 4), "\n")
cat("Events observed by 36 months =", sum(df[[event_var]] == 1 & df[[time_var]] <= time_horizon), "\n")
cat("Subjects with follow-up <=36 months and event=0, i.e. censored/administratively short before horizon =",
    sum(df[[event_var]] == 0 & df[[time_var]] <= time_horizon), "\n")
cat("Naive event fraction among subjects with fup<=36 months =",
    round(
      sum(df[[event_var]] == 1 & df[[time_var]] <= time_horizon) /
        sum(df[[time_var]] <= time_horizon),
      4
    ),
    "\n"
)
cat("Prediction horizon =", time_horizon, "months\n")
cat("Orientation =", MODEL_ORIENTATION, "\n")

X <- df |> dplyr::select(all_of(predictor_vars))
X_full <- stats::model.matrix(~ . - 1, data = X)
mm_cols <- colnames(X_full)

################################################################################
## 2) HELPER FUNCTIONS
################################################################################

clamp01 <- function(p) pmin(pmax(p, 1e-15), 1 - 1e-15)

safe_sd <- function(x) {
  if (sum(!is.na(x)) <= 1) return(NA_real_)
  stats::sd(x, na.rm = TRUE)
}

safe_q <- function(x, prob) {
  if (all(is.na(x))) return(NA_real_)
  as.numeric(stats::quantile(x, probs = prob, na.rm = TRUE, type = 7))
}

make_progress <- function(total) {
  utils::txtProgressBar(min = 0, max = total, style = 3, width = 50, char = "=")
}

ordinary_boot_idx_surv <- function(data) {
  sample(
    seq_len(nrow(data)),
    size = nrow(data),
    replace = TRUE
  )
}

make_mm <- function(newdata, template_cols) {
  mm <- stats::model.matrix(~ . - 1, data = newdata)
  miss <- setdiff(template_cols, colnames(mm))
  if (length(miss) > 0) {
    mm_miss <- matrix(0, nrow = nrow(mm), ncol = length(miss))
    colnames(mm_miss) <- miss
    mm <- cbind(mm, mm_miss)
  }
  mm[, template_cols, drop = FALSE]
}

################################################################################
## 3) IPCW DATA AT 36 MONTHS
################################################################################

get_ipcw_data <- function(data, horizon) {
  time <- data[[time_var]]
  event <- data[[event_var]]
  
  censor_fit <- survival::survfit(survival::Surv(time, 1 - event) ~ 1)
  
  G_at <- function(t) {
    s <- summary(censor_fit, times = t, extend = TRUE)$surv
    pmax(as.numeric(s), 1e-6)
  }
  
  y_h <- rep(NA_real_, length(time))
  w_h <- rep(0, length(time))
  
  event_before_h <- event == 1 & time <= horizon
  event_free_at_h <- time > horizon
  
  y_h[event_before_h] <- 1
  y_h[event_free_at_h] <- 0
  
  w_h[event_before_h] <- 1 / G_at(time[event_before_h])
  w_h[event_free_at_h] <- 1 / G_at(horizon)
  
  y_eval <- ifelse(is.na(y_h), 0, y_h)
  
  tibble::tibble(
    y_h = y_eval,
    w_h = w_h,
    known = !is.na(y_h),
    censored_before_horizon = is.na(y_h)
  )
}

weighted_auc_fast <- function(y, p, w) {
  ok <- is.finite(y) & is.finite(p) & is.finite(w) & w > 0
  y <- y[ok]
  p <- p[ok]
  w <- w[ok]
  
  if (length(unique(y)) < 2) return(NA_real_)
  
  ord <- order(p)
  y <- y[ord]
  p <- p[ord]
  w <- w[ord]
  
  tab <- tibble::tibble(y = y, p = p, w = w) |>
    dplyr::group_by(p) |>
    dplyr::summarise(
      w_case = sum(w[y == 1]),
      w_ctrl = sum(w[y == 0]),
      .groups = "drop"
    ) |>
    dplyr::arrange(p) |>
    dplyr::mutate(
      ctrl_less = dplyr::lag(cumsum(w_ctrl), default = 0),
      contribution = w_case * (ctrl_less + 0.5 * w_ctrl)
    )
  
  den <- sum(tab$w_case) * sum(tab$w_ctrl)
  if (den <= 0) return(NA_real_)
  
  sum(tab$contribution) / den
}

Brier_ipcw <- function(y, p, w, n_total) {
  ok <- is.finite(y) & is.finite(p) & is.finite(w)
  sum(w[ok] * (y[ok] - p[ok])^2) / n_total
}

LogLoss_ipcw <- function(y, p, w, n_total) {
  p <- clamp01(p)
  ok <- is.finite(y) & is.finite(p) & is.finite(w)
  -sum(w[ok] * (y[ok] * log(p[ok]) + (1 - y[ok]) * log(1 - p[ok]))) / n_total
}

CalSlope_ipcw <- function(y, p, w) {
  p <- clamp01(p)
  lp <- qlogis(p)
  ok <- is.finite(y) & is.finite(lp) & is.finite(w) & w > 0
  y <- y[ok]
  lp <- lp[ok]
  w <- w[ok]
  if (length(unique(y)) < 2) return(NA_real_)
  fit <- try(stats::glm(y ~ lp, family = stats::binomial(), weights = w), silent = TRUE)
  if (inherits(fit, "try-error")) return(NA_real_)
  unname(stats::coef(fit)[2])
}

CalIntercept_ipcw <- function(y, p, w) {
  p <- clamp01(p)
  lp <- qlogis(p)
  ok <- is.finite(y) & is.finite(lp) & is.finite(w) & w > 0
  y <- y[ok]
  lp <- lp[ok]
  w <- w[ok]
  if (length(unique(y)) < 2) return(NA_real_)
  fit <- try(stats::glm(y ~ offset(lp), family = stats::binomial(), weights = w), silent = TRUE)
  if (inherits(fit, "try-error")) return(NA_real_)
  unname(stats::coef(fit)[1])
}

metric_vec_surv <- function(data, p, horizon) {
  ip <- get_ipcw_data(data, horizon)
  tibble::tibble(
    AUC = weighted_auc_fast(ip$y_h, p, ip$w_h),
    Brier = Brier_ipcw(ip$y_h, p, ip$w_h, n_total = nrow(data)),
    LogLoss = LogLoss_ipcw(ip$y_h, p, ip$w_h, n_total = nrow(data)),
    CalIntercept = CalIntercept_ipcw(ip$y_h, p, ip$w_h),
    CalSlope = CalSlope_ipcw(ip$y_h, p, ip$w_h)
  )
}

## Probability-oriented tuning score for RSF. Higher is better.
## Primary and only tuning target: low IPCW-adjusted logarithmic loss at 36 months.
## Brier score and calibration metrics are retained for reporting, not for RSF model selection.
score_for_tuning_probability <- function(metrics) {
  -metrics$LogLoss
}

ip_full <- get_ipcw_data(df, time_horizon)
cat("IPCW estimated event risk at 36 months =",
    round(sum(ip_full$w_h * ip_full$y_h) / nrow(df), 4),
    "\n")
cat("Known outcome status at 36 months =",
    sum(ip_full$known),
    "of",
    nrow(df),
    "\n")

################################################################################
## 4) COX PH MODEL AND DIAGNOSTICS
################################################################################

cox_ph <- survival::coxph(
  survival::Surv(fup_D_IMA_STROKE, death_IMA_Stroke) ~
    age_at_LDL_index + ANA_SESSO + PAS + colesterolo_mmol +
    HDL_mmol + diab_index + FR_FUMO,
  data = df,
  x = TRUE,
  y = TRUE
)

print(summary(cox_ph))

ph_clinical <- survival::cox.zph(cox_ph)
print(ph_clinical)
plot(ph_clinical)

ph_table <- data.frame(
  Variable = rownames(ph_clinical$table),
  ph_clinical$table,
  row.names = NULL
)

extract_smooth_effect <- function(ph_object, variable_name, n_grid = 100) {
  y <- ph_object$y[, variable_name]
  x <- ph_object$x
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]
  y <- y[ok]
  sm <- stats::smooth.spline(x = x, y = y)
  grid_time <- seq(min(x), max(x), length.out = n_grid)
  pred <- stats::predict(sm, x = grid_time)
  tibble::tibble(
    variable = variable_name,
    time = pred$x,
    beta_t = pred$y,
    HR_t = exp(pred$y)
  )
}

effects_time <- purrr::map_dfr(
  colnames(ph_clinical$y),
  ~ extract_smooth_effect(ph_clinical, .x)
)

effect_summary <- effects_time |>
  dplyr::group_by(variable) |>
  dplyr::summarise(
    beta_min = min(beta_t, na.rm = TRUE),
    beta_max = max(beta_t, na.rm = TRUE),
    beta_range = beta_max - beta_min,
    HR_min = min(HR_t, na.rm = TRUE),
    HR_max = max(HR_t, na.rm = TRUE),
    HR_ratio_max_min = HR_max / HR_min,
    .groups = "drop"
  )

################################################################################
## 5) RANDOM SURVIVAL FOREST TUNING — PROBABILITY ORIENTED
################################################################################

## Probability-oriented RSF tuning uses the IPCW-adjusted logarithmic loss at
## 36 months as the sole model-selection criterion. Brier score and calibration
## are evaluated after model selection, not included in the tuning objective.
rf_grid <- expand.grid(
  mtry = c(2, 3),
  min.node.size = c(75, 125, 200),
  sample.fraction = c(0.8, 1.0),
  splitrule = "logrank"
)

cat("Tuning Random Survival Forest for probabilistic accuracy over", nrow(rf_grid), "combinations...\n")

rf_pb <- make_progress(nrow(rf_grid))
rf_tuning_list <- vector("list", nrow(rf_grid))

for (i in seq_len(nrow(rf_grid))) {
  utils::setTxtProgressBar(rf_pb, i)
  pars <- rf_grid[i, ]
  
  fit <- ranger::ranger(
    survival::Surv(fup_D_IMA_STROKE, death_IMA_Stroke) ~
      age_at_LDL_index + ANA_SESSO + PAS + colesterolo_mmol +
      HDL_mmol + diab_index + FR_FUMO,
    data = df,
    num.trees = rf_tune_trees,
    mtry = pars$mtry,
    min.node.size = pars$min.node.size,
    sample.fraction = pars$sample.fraction,
    splitrule = as.character(pars$splitrule),
    importance = "none",
    respect.unordered.factors = "order",
    seed = 123
  )
  
  p_oob <- try({
    idx_t <- which.min(abs(fit$unique.death.times - time_horizon))
    clamp01(1 - fit$survival[, idx_t])
  }, silent = TRUE)
  
  met <- if (!inherits(p_oob, "try-error")) {
    metric_vec_surv(df, p_oob, time_horizon)
  } else {
    tibble::tibble(AUC = NA, Brier = NA, LogLoss = NA,
                   CalIntercept = NA, CalSlope = NA)
  }
  
  rf_tuning_list[[i]] <- tibble::tibble(
    mtry = pars$mtry,
    min.node.size = pars$min.node.size,
    sample.fraction = pars$sample.fraction,
    splitrule = as.character(pars$splitrule),
    AUC = met$AUC,
    Brier = met$Brier,
    LogLoss = met$LogLoss,
    CalIntercept = met$CalIntercept,
    CalSlope = met$CalSlope,
    tuning_target = "min_IPCW_LogLoss",
    tuning_score = score_for_tuning_probability(met)
  )
}

close(rf_pb)
cat("\n")

rf_tuning <- dplyr::bind_rows(rf_tuning_list)

rf_best <- rf_tuning |>
  dplyr::arrange(dplyr::desc(tuning_score)) |>
  dplyr::slice_head(n = 1)

print(rf_tuning)
print(rf_best)

rf_best_mtry <- as.integer(rf_best$mtry[[1]])
rf_best_min_node <- as.integer(rf_best$min.node.size[[1]])
rf_best_sample_fraction <- as.numeric(rf_best$sample.fraction[[1]])

################################################################################
## 6) MODEL FIT AND PREDICTION FUNCTIONS
################################################################################

fit_cox_ph <- function(idx) {
  survival::coxph(
    survival::Surv(fup_D_IMA_STROKE, death_IMA_Stroke) ~
      age_at_LDL_index + ANA_SESSO + PAS + colesterolo_mmol +
      HDL_mmol + diab_index + FR_FUMO,
    data = df[idx, , drop = FALSE],
    x = TRUE,
    y = TRUE
  )
}

pred_cox_ph <- function(m, newdata) {
  as.numeric(riskRegression::predictRisk(
    object = m,
    newdata = newdata,
    times = time_horizon
  )[, 1])
}

fit_rf_surv <- function(idx) {
  ranger::ranger(
    survival::Surv(fup_D_IMA_STROKE, death_IMA_Stroke) ~
      age_at_LDL_index + ANA_SESSO + PAS + colesterolo_mmol +
      HDL_mmol + diab_index + FR_FUMO,
    data = df[idx, , drop = FALSE],
    num.trees = rf_final_trees,
    mtry = rf_best_mtry,
    min.node.size = rf_best_min_node,
    sample.fraction = rf_best_sample_fraction,
    splitrule = "logrank",
    importance = "none",
    respect.unordered.factors = "order",
    seed = 123
  )
}

pred_rf_surv <- function(m, newdata) {
  pr <- stats::predict(m, data = newdata)
  idx_t <- which.min(abs(pr$unique.death.times - time_horizon))
  as.numeric(clamp01(1 - pr$survival[, idx_t]))
}

models <- list(
  Cox_PH = list(fit = fit_cox_ph, pred = pred_cox_ph),
  RF_survival = list(fit = fit_rf_surv, pred = pred_rf_surv)
)

################################################################################


################################################################################
## 7) SHARED ORDINARY BOOTSTRAP
##
## The same subject-level bootstrap resamples are used:
##   - for Cox PH and Random Survival Forest;
##   - for optimism-corrected performance;
##   - for individual prediction uncertainty / decision fragility;
##   - for optimism-corrected IPCW decision curve analysis.
##
## Each model is fitted only ONCE within each bootstrap replication.
################################################################################

set.seed(20260820)

bootstrap_indices <- replicate(
  B,
  ordinary_boot_idx_surv(df),
  simplify = FALSE
)

bootstrap_event_fraction <- purrr::map_dbl(
  bootstrap_indices,
  ~ mean(df[[event_var]][.x] == 1)
)

bootstrap_event36_fraction <- purrr::map_dbl(
  bootstrap_indices,
  ~ mean(
    df[[event_var]][.x] == 1 &
      df[[time_var]][.x] <= time_horizon
  )
)

cat(
  "Shared ordinary bootstrap generated:", B, "resamples\n",
  "Overall event fraction across resamples:",
  round(min(bootstrap_event_fraction), 4), "-",
  round(max(bootstrap_event_fraction), 4), "\n",
  "Event-by-36-month fraction across resamples:",
  round(min(bootstrap_event36_fraction), 4), "-",
  round(max(bootstrap_event36_fraction), 4), "\n"
)


################################################################################
## 8) CALIBRATION INSTABILITY
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
        stats::median(p, na.rm = TRUE),
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


calibration_instability_data <- function(apparent_pred, pred_boot, data, g = 10) {
  ip <- get_ipcw_data(data, time_horizon)
  breaks <- make_breaks_deciles(apparent_pred, g = g)

  apparent_df <- tibble::tibble(
    y = ip$y_h,
    w = ip$w_h,
    p = apparent_pred
  ) |>
    dplyr::filter(is.finite(y), is.finite(w), w > 0) |>
    dplyr::mutate(g = assign_groups(p, breaks)) |>
    dplyr::group_by(g) |>
    dplyr::summarise(
      mean_pred = stats::weighted.mean(p, w, na.rm = TRUE),
      obs = stats::weighted.mean(y, w, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(Type = "Apparent", bootstrap = NA_integer_)

  boot_df <- purrr::map_dfr(
    seq_len(ncol(pred_boot)),
    function(j) {
      tibble::tibble(
        y = ip$y_h,
        w = ip$w_h,
        p = pred_boot[, j]
      ) |>
        dplyr::filter(is.finite(y), is.finite(w), w > 0) |>
        dplyr::mutate(g = assign_groups(p, breaks)) |>
        dplyr::group_by(g) |>
        dplyr::summarise(
          mean_pred = stats::weighted.mean(p, w, na.rm = TRUE),
          obs = stats::weighted.mean(y, w, na.rm = TRUE),
          .groups = "drop"
        ) |>
        dplyr::mutate(Type = "Bootstrap", bootstrap = j)
    }
  )

  dplyr::bind_rows(apparent_df, boot_df)
}


plot_calibration_instability <- function(cal_df, model_name) {
  ggplot2::ggplot() +
    ggplot2::geom_line(
      data = cal_df |> dplyr::filter(Type == "Bootstrap"),
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
      data = cal_df |> dplyr::filter(Type == "Apparent"),
      ggplot2::aes(x = mean_pred, y = obs),
      size = 2
    ) +
    ggplot2::geom_line(
      data = cal_df |> dplyr::filter(Type == "Apparent"),
      ggplot2::aes(x = mean_pred, y = obs),
      linewidth = 0.9
    ) +
    ggplot2::geom_abline(
      intercept = 0,
      slope = 1,
      linetype = 2,
      color = "red"
    ) +
    ggplot2::coord_equal() +
    ggplot2::theme_bw() +
    ggplot2::scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.2)
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.2)
    ) +
    ggplot2::labs(
      title = model_name,
      x = paste0("Predicted risk at ", time_horizon, " months"),
      y = paste0("IPCW observed risk at ", time_horizon, " months")
    )
}


################################################################################
## 9) PATIENT-LEVEL PREDICTION UNCERTAINTY AND DECISION FRAGILITY
################################################################################

summarise_patient_bootstrap <- function(
    pred_boot,
    apparent_pred,
    data,
    model_name) {

  tibble::tibble(
    patient_id = seq_len(nrow(data)),
    time = data[[time_var]],
    event = data[[event_var]],
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


compute_fragility_grid <- function(
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
        n_above = sum(
          patient_tbl$pred_boot_p025 > thr,
          na.rm = TRUE
        ),
        n_below = sum(
          patient_tbl$pred_boot_p975 < thr,
          na.rm = TRUE
        )
      )
    }
  )
}


plot_fragility_vs_threshold <- function(fragility_grid_all) {
  ggplot2::ggplot(
    fragility_grid_all,
    ggplot2::aes(
      x = threshold,
      y = prop_fragile,
      color = Model,
      group = Model
    )
  ) +
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::geom_point(size = 2) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "Decision fragility vs threshold",
      x = "Threshold probability",
      y = "Proportion with fragile decision",
      color = ""
    )
}


################################################################################
## 10) IPCW DECISION CURVE ANALYSIS
################################################################################

decision_curve_data_ipcw <- function(
    data,
    p,
    thresholds = seq(0.01, 0.99, by = 0.01)) {

  ip <- get_ipcw_data(data, time_horizon)
  y_ipcw <- ip$y_h
  w_ipcw <- ip$w_h
  n_local <- nrow(data)

  event_risk_ipcw <-
    sum(w_ipcw * y_ipcw) /
    n_local

  purrr::map_dfr(
    thresholds,
    function(pt) {

      high <- p >= pt

      weighted_TP <-
        sum(
          w_ipcw *
            high *
            (y_ipcw == 1)
        )

      weighted_FP <-
        sum(
          w_ipcw *
            high *
            (y_ipcw == 0)
        )

      NB_model <-
        (weighted_TP / n_local) -
        (weighted_FP / n_local) *
        (pt / (1 - pt))

      NB_all <-
        event_risk_ipcw -
        (1 - event_risk_ipcw) *
        (pt / (1 - pt))

      tibble::tibble(
        threshold = pt,
        n_high = sum(high, na.rm = TRUE),
        prop_high = mean(high, na.rm = TRUE),
        weighted_TP = weighted_TP,
        weighted_FP = weighted_FP,
        event_risk_ipcw = event_risk_ipcw,
        NB_model = NB_model,
        NB_all = NB_all,
        NB_none = 0
      )
    }
  )
}


make_dca_long <- function(dca_df, model_name) {
  dplyr::bind_rows(
    dca_df |>
      dplyr::transmute(
        model = model_name,
        threshold,
        strategy = "Model apparent",
        NB = NB_apparent
      ),
    dca_df |>
      dplyr::transmute(
        model = model_name,
        threshold,
        strategy = "Model optimism-corrected",
        NB = NB_corrected
      ),
    dca_df |>
      dplyr::transmute(
        model = model_name,
        threshold,
        strategy = "Treat all",
        NB = NB_all
      ),
    dca_df |>
      dplyr::transmute(
        model = model_name,
        threshold,
        strategy = "Treat none",
        NB = NB_none
      )
  )
}


plot_dca <- function(dca_long_all) {
  ggplot2::ggplot(
    dca_long_all,
    ggplot2::aes(
      x = threshold,
      y = NB,
      color = strategy
    )
  ) +
    ggplot2::geom_line(linewidth = 0.9) +
    ggplot2::scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, by = 0.2)
    ) +
    ggplot2::coord_cartesian(
      ylim = c(-0.05, 0.20)
    ) +
    ggplot2::facet_wrap(~model) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      strip.text = ggplot2::element_text(size = 16),
      legend.position = "bottom",
      legend.text = ggplot2::element_text(size = 14),
      axis.text = ggplot2::element_text(size = 14),
      axis.title = ggplot2::element_text(size = 16)
    ) +
    ggplot2::labs(
      x = "Threshold probability",
      y = "Net benefit",
      color = ""
    )
}

## 11) IPCW LOG-LOSS PERMUTATION IMPORTANCE
################################################################################

permutation_importance_logloss <- function(model, pred_fun, data, model_name, n_perm = 5) {
  cat("Permutation importance IPCW-logloss for", model_name, "\n")
  
  p_ref <- clamp01(pred_fun(model, data))
  ip <- get_ipcw_data(data, time_horizon)
  base_logloss <- LogLoss_ipcw(ip$y_h, p_ref, ip$w_h, n_total = nrow(data))
  vars <- predictor_vars
  
  pb <- make_progress(length(vars))
  on.exit(try(close(pb), silent = TRUE), add = TRUE)
  
  out <- purrr::map_dfr(seq_along(vars), function(j) {
    utils::setTxtProgressBar(pb, j)
    v <- vars[j]
    increases <- rep(NA_real_, n_perm)
    
    for (k in seq_len(n_perm)) {
      data_perm <- data
      data_perm[[v]] <- sample(data_perm[[v]], replace = FALSE)
      p_perm <- try(clamp01(pred_fun(model, data_perm)), silent = TRUE)
      if (inherits(p_perm, "try-error")) next
      ll_perm <- LogLoss_ipcw(ip$y_h, p_perm, ip$w_h, n_total = nrow(data))
      increases[k] <- ll_perm - base_logloss
    }
    
    tibble::tibble(
      Model = model_name,
      Feature = v,
      base_logloss = base_logloss,
      mean_increase_logloss = mean(increases, na.rm = TRUE),
      sd_increase_logloss = stats::sd(increases, na.rm = TRUE),
      q025_increase_logloss = safe_q(increases, 0.025),
      q975_increase_logloss = safe_q(increases, 0.975),
      n_valid_perm = sum(!is.na(increases))
    )
  })
  
  cat("\n")
  
  out |> dplyr::arrange(dplyr::desc(mean_increase_logloss))
}

plot_importance_logloss <- function(imp_tbl, model_name, top_n = 15) {
  pdat <- imp_tbl |>
    dplyr::slice_head(n = top_n) |>
    dplyr::mutate(Feature = factor(Feature, levels = rev(Feature)))
  
  ggplot2::ggplot(pdat, ggplot2::aes(x = Feature, y = mean_increase_logloss)) +
    ggplot2::geom_col() +
    ggplot2::coord_flip() +
    ggplot2::theme_minimal() +
    ggplot2::labs(title = model_name, x = "", y = "Mean increase in IPCW logloss")
}

################################################################################

################################################################################
## 12) SHARED-BOOTSTRAP PIPELINE FOR ONE MODEL
##
## Within bootstrap replication b:
##   1. fit the model once on bootstrap sample b;
##   2. predict the original sample once;
##   3. obtain bootstrap-sample predictions by indexing those predictions;
##   4. use the same fit/predictions for performance, uncertainty and DCA.
################################################################################

run_model_pipeline_surv <- function(
    model_name,
    fit_fun,
    pred_fun,
    data,
    bootstrap_indices,
    thresholds_dca,
    thresholds_interest,
    n_perm_importance = 5) {

  cat("\n====================================================\n")
  cat("MODEL:", model_name, "\n")
  cat("====================================================\n")

  n_local <- nrow(data)
  B_local <- length(bootstrap_indices)

  ##########################################################################
  ## Apparent model
  ##########################################################################

  apparent_model <- fit_fun(seq_len(n_local))

  apparent_pred <- clamp01(
    pred_fun(
      apparent_model,
      data
    )
  )

  apparent_metrics <-
    metric_vec_surv(
      data,
      apparent_pred,
      time_horizon
    )

  apparent_dca <-
    decision_curve_data_ipcw(
      data,
      apparent_pred,
      thresholds_dca
    ) |>
    dplyr::transmute(
      threshold,
      n_high_apparent = n_high,
      prop_high_apparent = prop_high,
      weighted_TP_apparent = weighted_TP,
      weighted_FP_apparent = weighted_FP,
      event_risk_ipcw,
      NB_apparent = NB_model,
      NB_all,
      NB_none
    )

  ##########################################################################
  ## Storage
  ##########################################################################

  pred_boot <- matrix(
    NA_real_,
    nrow = n_local,
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

  valid <- logical(B_local)

  fail_log <- tibble::tibble(
    bootstrap = integer(),
    stage = character(),
    message = character()
  )

  ##########################################################################
  ## One bootstrap loop
  ##########################################################################

  pb <- make_progress(B_local)

  on.exit(
    try(close(pb), silent = TRUE),
    add = TRUE
  )

  for (b in seq_len(B_local)) {

    utils::setTxtProgressBar(pb, b)

    idxb <- bootstrap_indices[[b]]

    model_b <- try(
      fit_fun(idxb),
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

    ## Predict original individuals once.
    p_original <- try(
      clamp01(
        pred_fun(
          model_b,
          data
        )
      ),
      silent = TRUE
    )

    if (
      inherits(p_original, "try-error") ||
      length(p_original) != n_local ||
      any(!is.finite(p_original))
    ) {
      fail_log <- dplyr::bind_rows(
        fail_log,
        tibble::tibble(
          bootstrap = b,
          stage = "predict_original",
          message = "Prediction failed, had wrong length, or returned non-finite values."
        )
      )
      next
    }

    ## Because bootstrap observations are duplicated original rows,
    ## these are exactly the in-bootstrap predictions from the same fitted model.
    p_bootsample <- p_original[idxb]

    ## Patient-level prediction distribution
    pred_boot[, b] <- p_original

    ## Performance in bootstrap sample and on original sample
    perf_boot <-
      metric_vec_surv(
        data[idxb, , drop = FALSE],
        p_bootsample,
        time_horizon
      )

    perf_test <-
      metric_vec_surv(
        data,
        p_original,
        time_horizon
      )

    performance_list[[b]] <-
      dplyr::bind_cols(
        tibble::tibble(
          bootstrap = b,
          bootstrap_event_fraction =
            mean(
              data[[event_var]][idxb] == 1
            ),
          bootstrap_event36_fraction =
            mean(
              data[[event_var]][idxb] == 1 &
                data[[time_var]][idxb] <= time_horizon
            )
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

    ## DCA from the same bootstrap fit
    dca_boot <-
      decision_curve_data_ipcw(
        data[idxb, , drop = FALSE],
        p_bootsample,
        thresholds_dca
      ) |>
      dplyr::select(
        threshold,
        NB_boot = NB_model
      )

    dca_test <-
      decision_curve_data_ipcw(
        data,
        p_original,
        thresholds_dca
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
        NB_optimism =
          NB_boot -
          NB_test,
        .before = 1
      )

    valid[b] <- TRUE
  }

  cat("\n")

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

  valid_bootstrap_ids <- which(valid)

  colnames(pred_boot) <-
    paste0(
      "boot_",
      valid_bootstrap_ids
    )

  performance_boot <-
    dplyr::bind_rows(
      performance_list
    )

  dca_boot_detail <-
    dplyr::bind_rows(
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
  ## Optimism-corrected performance
  ##########################################################################

  performance_optimism <-
    performance_boot |>
    dplyr::transmute(
      bootstrap,
      AUC_optimism =
        AUC_boot - AUC_test,
      Brier_optimism =
        Brier_boot - Brier_test,
      LogLoss_optimism =
        LogLoss_boot - LogLoss_test,
      CalIntercept_optimism =
        CalIntercept_boot - CalIntercept_test,
      CalSlope_optimism =
        CalSlope_boot - CalSlope_test
    )

  performance_corrected_dist <-
    performance_optimism |>
    dplyr::transmute(
      bootstrap,
      AUC_corrected_iter =
        apparent_metrics$AUC -
        AUC_optimism,
      Brier_corrected_iter =
        apparent_metrics$Brier -
        Brier_optimism,
      LogLoss_corrected_iter =
        apparent_metrics$LogLoss -
        LogLoss_optimism,
      CalIntercept_corrected_iter =
        apparent_metrics$CalIntercept -
        CalIntercept_optimism,
      CalSlope_corrected_iter =
        apparent_metrics$CalSlope -
        CalSlope_optimism
    )

  performance_summary <-
    tibble::tibble(
      Model = model_name,

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

      LogLoss_apparent =
        apparent_metrics$LogLoss,
      LogLoss_corrected =
        apparent_metrics$LogLoss -
        mean(
          performance_optimism$LogLoss_optimism,
          na.rm = TRUE
        ),
      LogLoss_p025 =
        safe_q(
          performance_corrected_dist$LogLoss_corrected_iter,
          0.025
        ),
      LogLoss_p975 =
        safe_q(
          performance_corrected_dist$LogLoss_corrected_iter,
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
  ## Patient-level uncertainty and fragility
  ##########################################################################

  patient_tbl <-
    summarise_patient_bootstrap(
      pred_boot = pred_boot,
      apparent_pred = apparent_pred,
      data = data,
      model_name = model_name
    )

  fragility_grid <-
    compute_fragility_grid(
      patient_tbl = patient_tbl,
      thresholds = thresholds_interest,
      model_name = model_name
    )

  calibration_df <-
    calibration_instability_data(
      apparent_pred = apparent_pred,
      pred_boot = pred_boot,
      data = data,
      g = 10
    )

  ##########################################################################
  ## Optimism-corrected DCA
  ##########################################################################

  dca_optimism <-
    dca_boot_detail |>
    dplyr::group_by(threshold) |>
    dplyr::summarise(
      mean_NB_optimism =
        mean(
          NB_optimism,
          na.rm = TRUE
        ),
      n_valid =
        sum(
          !is.na(NB_optimism)
        ),
      .groups = "drop"
    )

  dca_df <-
    apparent_dca |>
    dplyr::left_join(
      dca_optimism,
      by = "threshold"
    ) |>
    dplyr::mutate(
      NB_corrected =
        NB_apparent -
        mean_NB_optimism
    )

  dca_long <-
    make_dca_long(
      dca_df,
      model_name
    )

  ##########################################################################
  ## Permutation importance on apparent model
  ##########################################################################

  importance <-
    permutation_importance_logloss(
      model = apparent_model,
      pred_fun = pred_fun,
      data = data,
      model_name = model_name,
      n_perm = n_perm_importance
    )

  list(
    apparent_model =
      apparent_model,
    apparent_pred =
      apparent_pred,

    perf_summary =
      performance_summary,
    perf_raw_boot =
      performance_boot,
    perf_optimism_detail =
      performance_optimism,
    perf_corrected_dist =
      performance_corrected_dist,

    pred_boot =
      pred_boot,
    patient_tbl =
      patient_tbl,
    fragility_grid =
      fragility_grid,
    calibration_df =
      calibration_df,

    dca_df =
      dca_df,
    dca_long =
      dca_long,
    dca_boot_detail =
      dca_boot_detail,

    importance_logloss =
      importance,

    fail_log =
      fail_log,
    valid_bootstrap_ids =
      valid_bootstrap_ids
  )
}


################################################################################
## 13) RUN BOTH MODELS
################################################################################

cat(
  "Bootstrap settings: B =", B,
  "| ordinary subject-level bootstrap",
  "| shared across models and analyses",
  "| n_perm_importance =", n_perm_importance,
  "\n"
)

all_results <- list()

for (model_name in names(models)) {

  all_results[[model_name]] <-
    run_model_pipeline_surv(
      model_name = model_name,
      fit_fun = models[[model_name]]$fit,
      pred_fun = models[[model_name]]$pred,
      data = df,
      bootstrap_indices = bootstrap_indices,
      thresholds_dca = thresholds_dca,
      thresholds_interest = thresholds_interest,
      n_perm_importance = n_perm_importance
    )
}


################################################################################
## 14) SUMMARY TABLES
################################################################################

perf_summary <-
  dplyr::bind_rows(
    lapply(
      all_results,
      `[[`,
      "perf_summary"
    )
  )

fragility_grid_all <-
  dplyr::bind_rows(
    lapply(
      all_results,
      `[[`,
      "fragility_grid"
    )
  ) |>
  dplyr::mutate(
    Fragility_percent =
      100 * prop_fragile
  )

patient_tbl_all <-
  dplyr::bind_rows(
    lapply(
      all_results,
      `[[`,
      "patient_tbl"
    )
  )

dca_long_all <-
  dplyr::bind_rows(
    lapply(
      all_results,
      `[[`,
      "dca_long"
    )
  )

importance_logloss_all <-
  dplyr::bind_rows(
    lapply(
      all_results,
      `[[`,
      "importance_logloss"
    )
  )

perf_optimism_detail_all <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(mn) {
        all_results[[mn]]$perf_optimism_detail |>
          dplyr::mutate(
            Model = mn,
            .before = 1
          )
      }
    )
  )

perf_corrected_dist_all <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(mn) {
        all_results[[mn]]$perf_corrected_dist |>
          dplyr::mutate(
            Model = mn,
            .before = 1
          )
      }
    )
  )

dca_boot_detail_all <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(mn) {
        all_results[[mn]]$dca_boot_detail |>
          dplyr::mutate(
            Model = mn,
            .before = 1
          )
      }
    )
  )

fail_log_all <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(mn) {
        all_results[[mn]]$fail_log |>
          dplyr::mutate(
            Model = mn,
            .before = 1
          )
      }
    )
  )

model_uncertainty_summary <-
  patient_tbl_all |>
  dplyr::group_by(model) |>
  dplyr::summarise(
    n = dplyr::n(),
    mean_pred_sd =
      mean(
        pred_boot_sd,
        na.rm = TRUE
      ),
    median_pred_sd =
      stats::median(
        pred_boot_sd,
        na.rm = TRUE
      ),
    q90_pred_sd =
      as.numeric(
        stats::quantile(
          pred_boot_sd,
          0.90,
          na.rm = TRUE
        )
      ),
    q95_pred_sd =
      as.numeric(
        stats::quantile(
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
      stats::median(
        pred_boot_interval_width,
        na.rm = TRUE
      ),
    .groups = "drop"
  ) |>
  dplyr::rename(
    Model = model
  )

print(perf_summary)
print(fragility_grid_all)
print(model_uncertainty_summary)


################################################################################
## 15) DCA HIGH-THRESHOLD CHECK
################################################################################

dca_high_threshold_check <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(mn) {

        all_results[[mn]]$dca_df |>
          dplyr::filter(
            threshold >= 0.20
          ) |>
          dplyr::select(
            threshold,
            n_high_apparent,
            prop_high_apparent,
            weighted_TP_apparent,
            weighted_FP_apparent,
            NB_apparent,
            NB_corrected,
            NB_all,
            NB_none
          ) |>
          dplyr::mutate(
            Model = mn,
            .before = 1
          )
      }
    )
  )

print(dca_high_threshold_check)


################################################################################
## 16) PLOTS
################################################################################

p_cal_cox <-
  plot_calibration_instability(
    all_results$Cox_PH$calibration_df,
    "Cox PH"
  )

p_cal_rf <-
  plot_calibration_instability(
    all_results$RF_survival$calibration_df,
    "Random Survival Forest"
  )

p_cal_all <-
  ggpubr::ggarrange(
    p_cal_cox,
    p_cal_rf,
    ncol = 2
  )

print(p_cal_all)


p_dca <-
  plot_dca(
    dca_long_all
  )

print(p_dca)


p_fragility_grid <-
  plot_fragility_vs_threshold(
    fragility_grid_all
  )

print(p_fragility_grid)


p_imp_cox <-
  plot_importance_logloss(
    all_results$Cox_PH$importance_logloss,
    "Cox PH",
    top_n = 15
  )

p_imp_rf <-
  plot_importance_logloss(
    all_results$RF_survival$importance_logloss,
    "Random Survival Forest",
    top_n = 15
  )

p_imp_all <-
  ggpubr::ggarrange(
    p_imp_cox,
    p_imp_rf,
    ncol = 2
  )

print(p_imp_all)


################################################################################
## 17) DCA AND FRAGILITY TABLES AT PRESPECIFIED THRESHOLDS
################################################################################

dca_nb_table <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(mn) {

        all_results[[mn]]$dca_df |>
          dplyr::mutate(
            threshold_round =
              round(
                threshold,
                2
              )
          ) |>
          dplyr::filter(
            threshold_round %in%
              thresholds_interest
          ) |>
          dplyr::transmute(
            Model = mn,
            threshold =
              threshold_round,
            NB_apparent,
            NB_corrected,
            NB_all,
            NB_none
          )
      }
    )
  )

final_table <-
  dca_nb_table |>
  dplyr::left_join(
    fragility_grid_all |>
      dplyr::select(
        Model,
        threshold,
        n_fragile,
        Fragility_percent
      ),
    by = c(
      "Model",
      "threshold"
    )
  ) |>
  dplyr::arrange(
    Model,
    threshold
  )

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
    names_glue =
      "t={threshold}_{.value}"
  )

print(dca_nb_table)
print(final_table)
print(final_wide)


################################################################################
## 18) COX HAZARD-RATIO TABLE
################################################################################

extract_hr <- function(
    model,
    model_name) {

  s <- summary(model)

  tibble::tibble(
    Model = model_name,
    Variable =
      rownames(
        s$coefficients
      ),
    HR =
      s$coefficients[
        ,
        "exp(coef)"
      ],
    CI_lower =
      s$conf.int[
        ,
        "lower .95"
      ],
    CI_upper =
      s$conf.int[
        ,
        "upper .95"
      ],
    p_value =
      s$coefficients[
        ,
        "Pr(>|z|)"
      ]
  )
}

hr_cox_ph <-
  extract_hr(
    cox_ph,
    "Cox_PH"
  )

print(hr_cox_ph)


################################################################################
## 19) EXPORT RESULTS
################################################################################

calibration_data_all <-
  dplyr::bind_rows(
    lapply(
      names(all_results),
      function(mn) {

        all_results[[mn]]$calibration_df |>
          dplyr::mutate(
            Model = mn,
            .before = 1
          )
      }
    )
  )

bootstrap_design_summary <-
  tibble::tibble(
    bootstrap =
      seq_len(B),
    event_fraction =
      bootstrap_event_fraction,
    event_by_36m_fraction =
      bootstrap_event36_fraction
  )

writexl::write_xlsx(
  list(
    Settings =
      tibble::tibble(
        setting = c(
          "orientation",
          "n",
          "n_predictors",
          "time_horizon_months",
          "total_events",
          "crude_total_event_rate",
          "events_by_36_months",
          "naive_event_fraction_fup_le_36",
          "ipcw_event_risk_36_months",
          "bootstrap_type",
          "bootstrap_shared_across_models",
          "bootstrap_fit_reused_across_analyses",
          "B"
        ),
        value = c(
          MODEL_ORIENTATION,
          nrow(df),
          length(predictor_vars),
          time_horizon,
          sum(df[[event_var]] == 1),
          mean(df[[event_var]] == 1),
          sum(
            df[[event_var]] == 1 &
              df[[time_var]] <= time_horizon
          ),
          sum(
            df[[event_var]] == 1 &
              df[[time_var]] <= time_horizon
          ) /
            sum(
              df[[time_var]] <= time_horizon
            ),
          sum(
            ip_full$w_h *
              ip_full$y_h
          ) /
            nrow(df),
          "ordinary subject-level bootstrap",
          "TRUE",
          "TRUE",
          B
        )
      ),

    Bootstrap_design =
      bootstrap_design_summary,

    PH_test =
      ph_table,

    Schoenfeld_effect_summary =
      effect_summary,

    Cox_PH_HR =
      hr_cox_ph,

    RF_tuning =
      rf_tuning,

    RF_best =
      rf_best,

    Performance_apparent_corrected =
      perf_summary,

    Performance_optimism_detail =
      perf_optimism_detail_all,

    Performance_corrected_dist =
      perf_corrected_dist_all,

    Patient_uncertainty =
      patient_tbl_all,

    Model_uncertainty_summary =
      model_uncertainty_summary,

    Calibration_data =
      calibration_data_all,

    DCA_long =
      dca_long_all,

    DCA_boot_detail =
      dca_boot_detail_all,

    DCA_high_threshold_check =
      dca_high_threshold_check,

    Fragility_grid =
      fragility_grid_all,

    Final_DCA_fragility_table =
      final_table,

    Final_DCA_fragility_wide =
      final_wide,

    Importance_IPCW_logloss =
      importance_logloss_all,

    Fail_log =
      fail_log_all
  ),

  path =
    "survival_prediction_probability_calibration_RSF_IPCWlogloss_CoxPH_RSF_shared_ordinary_boot500.xlsx"
)

ggplot2::ggsave(
  "probability_calibration_RSF_IPCWlogloss_CoxPH_RSF_shared_ordinary_boot500_calibration_instability.png",
  p_cal_all,
  width = 12,
  height = 4,
  dpi = 300
)

ggplot2::ggsave(
  "probability_calibration_RSF_IPCWlogloss_CoxPH_RSF_shared_ordinary_boot500_dca_IPCW_36months.png",
  p_dca,
  width = 10,
  height = 6,
  dpi = 600
)

ggplot2::ggsave(
  "probability_calibration_RSF_IPCWlogloss_CoxPH_RSF_shared_ordinary_boot500_fragility.png",
  p_fragility_grid,
  width = 8,
  height = 5,
  dpi = 600
)

ggplot2::ggsave(
  "probability_calibration_RSF_IPCWlogloss_CoxPH_RSF_shared_ordinary_boot500_importance.png",
  p_imp_all,
  width = 12,
  height = 4,
  dpi = 300
)

################################################################################
## END
################################################################################
