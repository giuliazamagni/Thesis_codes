################################################################################
## CONFORMAL AND VENN–ABERS UNDER MISCALIBRATION
## Overconfidence, monotone recalibration, and limited calibration data
################################################################################

## ===============================
## 0. SETUP
## ===============================
library(ggplot2)
library(dplyr)
library(tidyr)

set.seed(123)

n_train <- 500
n_cal <- 100
n_test <- 100
n_cal_small <- 50
alpha <- 0.05

## ===============================
## 1. DATA-GENERATING MECHANISM
## ===============================

simulate_data <- function(n) {
  
  # True individual event probabilities
  true_p <- runif(n, 0.05, 0.95)
  
  # Binary outcomes
  y <- rbinom(n, 1, true_p)
  
  # Artificially overconfident probability predictions
  pred_p_over <- plogis(
    1.8 * qlogis(true_p) + 0.5
  )
  
  data.frame(
    y = y,
    true_p = true_p,
    pred_p_over = pred_p_over
  )
}

## ===============================
## 2. THREE INDEPENDENT SAMPLES
## ===============================

# Training/development set:
# used only to estimate logistic recalibration
set.seed(101)
Dtrain <- simulate_data(n_train)

# Conformal calibration set
set.seed(202)
Dcal <- simulate_data(n_cal)

# Independent test set
set.seed(303)
Dtest <- simulate_data(n_test)

stopifnot(
  nrow(Dtrain) == n_train,
  nrow(Dcal) == n_cal,
  nrow(Dtest) == n_test
)

## ===============================
## 3. LOGISTIC RECALIBRATION
## ===============================

# The recalibration function is estimated exclusively
# in the independent training/development sample.

fit_cal <- glm(
  y ~ qlogis(pred_p_over),
  data = Dtrain,
  family = binomial()
)

cal_intercept <- coef(fit_cal)[1]
cal_slope <- coef(fit_cal)[2]

recalibrate <- function(p) {
  plogis(
    cal_intercept +
      cal_slope * qlogis(p)
  )
}

Dtrain$pred_p_shr <- recalibrate(Dtrain$pred_p_over)
Dcal$pred_p_shr <- recalibrate(Dcal$pred_p_over)
Dtest$pred_p_shr <- recalibrate(Dtest$pred_p_over)

cat(
  "Recalibration intercept:",
  round(cal_intercept, 3),
  "\n"
)

cat(
  "Recalibration slope:",
  round(cal_slope, 3),
  "\n\n"
)

## ===============================
## 4. MONOTONICITY CHECK
## ===============================

# A positive calibration slope implies a strictly
# increasing transformation of the original score.

same_ranking_cal <- identical(
  order(Dcal$pred_p_over),
  order(Dcal$pred_p_shr)
)

same_ranking_test <- identical(
  order(Dtest$pred_p_over),
  order(Dtest$pred_p_shr)
)

cat(
  "Same ranking in calibration set:",
  same_ranking_cal,
  "\n"
)

cat(
  "Same ranking in test set:",
  same_ranking_test,
  "\n\n"
)

## ===============================
## 5. NON-CONFORMITY SCORES
## ===============================

nonconformity_score <- function(y, p) {
  1 - ifelse(
    y == 1,
    p,
    1 - p
  )
}

Dcal$ncs_over <- nonconformity_score(
  Dcal$y,
  Dcal$pred_p_over
)

Dcal$ncs_shr <- nonconformity_score(
  Dcal$y,
  Dcal$pred_p_shr
)

## ===============================
## 6. NCS SUMMARY
## ===============================

ncs_df <- bind_rows(
  data.frame(
    NCS = Dcal$ncs_over,
    Model = "Overconfident"
  ),
  data.frame(
    NCS = Dcal$ncs_shr,
    Model = "Shrinkage-corrected"
  )
)

ncs_df$Model <- factor(
  ncs_df$Model,
  levels = c(
    "Overconfident",
    "Shrinkage-corrected"
  )
)

ncs_summary <- ncs_df |>
  group_by(Model) |>
  summarise(
    n = n(),
    median_NCS = median(NCS),
    Q1_NCS = quantile(NCS, 0.25),
    Q3_NCS = quantile(NCS, 0.75),
    IQR_NCS = IQR(NCS),
    .groups = "drop"
  )

print(ncs_summary)

## ===============================
## 7. NCS DENSITY PLOT
## ===============================

p_ncs <- ggplot(
  ncs_df,
  aes(
    x = NCS,
    fill = Model
  )
) +
  geom_density(
    alpha = 0.55
  ) +
  scale_fill_brewer(
    palette = "Set3"
  ) +
  scale_x_continuous(
    limits = c(0, 1)
  ) +
  labs(
    x = "Non-conformity score",
    y = "Density",
    fill = ""
  ) +
  theme_bw(base_size = 13) +
  theme(
    legend.position = "bottom"
  )

print(p_ncs)

## ===============================
## 8. FINITE-SAMPLE CONFORMAL QUANTILE
## ===============================

# Let m denote the number of calibration observations.
#
# k = ceiling((m + 1)(1 - alpha))
#
# q = kth ordered non-conformity score
#
# tau = 1 - q
#
# q is the conformal quantile on the NCS scale.
# tau is the corresponding probability inclusion threshold.

conformal_quantile <- function(scores, alpha) {
  
  m <- length(scores)
  
  k <- ceiling(
    (m + 1) *
      (1 - alpha)
  )
  
  k <- min(k, m)
  
  list(
    k = k,
    q = sort(scores)[k]
  )
}

cq_over <- conformal_quantile(
  Dcal$ncs_over,
  alpha
)

cq_shr <- conformal_quantile(
  Dcal$ncs_shr,
  alpha
)

q_over <- cq_over$q
q_shr <- cq_shr$q

tau_over <- 1 - q_over
tau_shr <- 1 - q_shr

## ===============================
## 9. CONFORMAL PREDICTION SETS
## ===============================

# For each candidate class c:
#
# alpha_c = 1 - p_c
#
# Include c if:
#
# alpha_c <= q
#
# equivalently:
#
# p_c >= tau

C_over_1 <- Dtest$pred_p_over >= tau_over
C_over_0 <- (1 - Dtest$pred_p_over) >= tau_over

C_shr_1 <- Dtest$pred_p_shr >= tau_shr
C_shr_0 <- (1 - Dtest$pred_p_shr) >= tau_shr

## ===============================
## 10. COVERAGE AND EFFICIENCY
## ===============================

coverage_over <- mean(
  (Dtest$y == 1 & C_over_1) |
    (Dtest$y == 0 & C_over_0)
)

coverage_shr <- mean(
  (Dtest$y == 1 & C_shr_1) |
    (Dtest$y == 0 & C_shr_0)
)

# Size of each prediction set
set_size_over <- C_over_0 + C_over_1
set_size_shr <- C_shr_0 + C_shr_1

# Efficiency represented by singleton rate
singleton_over <- mean(
  set_size_over == 1
)

singleton_shr <- mean(
  set_size_shr == 1
)

# Mean prediction-set size: lower = more efficient
mean_set_size_over <- mean(
  set_size_over
)

mean_set_size_shr <- mean(
  set_size_shr
)

two_label_over <- mean(
  set_size_over == 2
)

two_label_shr <- mean(
  set_size_shr == 2
)

empty_over <- mean(
  set_size_over == 0
)

empty_shr <- mean(
  set_size_shr == 0
)

## ===============================
## 11. CONFORMAL SUMMARY TABLE
## ===============================

conformal_summary <- data.frame(
  Model = c(
    "Overconfident",
    "Shrinkage-corrected"
  ),
  
  alpha = alpha,
  
  Calibration_n = n_cal,
  
  Conformal_order_k = c(
    cq_over$k,
    cq_shr$k
  ),
  
  Conformal_quantile_q = c(
    q_over,
    q_shr
  ),
  
  Inclusion_threshold_tau = c(
    tau_over,
    tau_shr
  ),
  
  Coverage = c(
    coverage_over,
    coverage_shr
  ),
  
  Efficiency_singleton_rate = c(
    singleton_over,
    singleton_shr
  ),
  
  Mean_prediction_set_size = c(
    mean_set_size_over,
    mean_set_size_shr
  ),
  
  Two_label_rate = c(
    two_label_over,
    two_label_shr
  ),
  
  Empty_set_rate = c(
    empty_over,
    empty_shr
  )
)

table_conformal_ncs <- conformal_summary |>
  left_join(
    ncs_summary,
    by = "Model"
  )

print(table_conformal_ncs)

## ===============================
## 12. PRINT CONFORMAL RULES
## ===============================

print_conformal_rule <- function(
    model_name,
    q,
    tau
) {
  
  cat(
    "\n",
    model_name,
    "\n",
    "q = ",
    round(q, 3),
    "\n",
    "tau = 1 - q = ",
    round(tau, 3),
    "\n",
    sep = ""
  )
  
  if (tau <= 0.5) {
    
    cat(
      "{0}   if p < ",
      round(tau, 3),
      "\n",
      "{0,1} if ",
      round(tau, 3),
      " <= p <= ",
      round(1 - tau, 3),
      "\n",
      "{1}   if p > ",
      round(1 - tau, 3),
      "\n",
      sep = ""
    )
    
  } else {
    
    cat(
      "{0} if p <= ",
      round(1 - tau, 3),
      "\n",
      "{}  if ",
      round(1 - tau, 3),
      " < p < ",
      round(tau, 3),
      "\n",
      "{1} if p >= ",
      round(tau, 3),
      "\n",
      sep = ""
    )
  }
}

print_conformal_rule(
  "OVERCONFIDENT MODEL",
  q_over,
  tau_over
)

print_conformal_rule(
  "SHRINKAGE-CORRECTED MODEL",
  q_shr,
  tau_shr
)

## ===============================
## 13. LIMITED VA CALIBRATION SET
## ===============================

# CS = 50 is nested within CS = 100.
# The prediction model, recalibration function,
# and test set remain unchanged.

set.seed(456)

idx_cal_small <- sample(
  seq_len(nrow(Dcal)),
  size = n_cal_small,
  replace = FALSE
)

Dcal_small <- Dcal[
  idx_cal_small,
  ,
  drop = FALSE
]

stopifnot(
  nrow(Dcal_small) == n_cal_small
)

## ===============================
## 14. VENN–ABERS FUNCTION
## ===============================

# For each test score, construct two augmented
# calibration samples by assigning the hypothetical
# test label first to 0 and then to 1.
#
# The isotonic fitted probability corresponding to
# the appended test observation is extracted directly.

va_predict <- function(
    scores_cal,
    y_cal,
    score_new
) {
  
  n_local <- length(scores_cal)
  
  fit_hypothetical <- function(y_new) {
    
    scores_aug <- c(
      scores_cal,
      score_new
    )
    
    y_aug <- c(
      y_cal,
      y_new
    )
    
    # Continuous scores make exact ties extremely unlikely.
    # Secondary ordering ensures deterministic behavior.
    ord <- order(
      scores_aug,
      seq_along(scores_aug)
    )
    
    iso <- isoreg(
      scores_aug[ord],
      y_aug[ord]
    )
    
    test_position <- which(
      ord == n_local + 1L
    )
    
    iso$yf[test_position]
  }
  
  p0 <- fit_hypothetical(0)
  p1 <- fit_hypothetical(1)
  
  c(
    lower = min(p0, p1),
    upper = max(p0, p1)
  )
}

## ===============================
## 15. VA: OVERCONFIDENT, CS = 100
## ===============================

va_over <- t(
  vapply(
    Dtest$pred_p_over,
    function(s) {
      va_predict(
        scores_cal = Dcal$pred_p_over,
        y_cal = Dcal$y,
        score_new = s
      )
    },
    numeric(2)
  )
)

Dtest$va_over_lower <- va_over[, "lower"]
Dtest$va_over_upper <- va_over[, "upper"]

Dtest$width_over <-
  Dtest$va_over_upper -
  Dtest$va_over_lower

## ===============================
## 16. VA: SHRINKAGE, CS = 100
## ===============================

va_shr <- t(
  vapply(
    Dtest$pred_p_shr,
    function(s) {
      va_predict(
        scores_cal = Dcal$pred_p_shr,
        y_cal = Dcal$y,
        score_new = s
      )
    },
    numeric(2)
  )
)

Dtest$va_shr_lower <- va_shr[, "lower"]
Dtest$va_shr_upper <- va_shr[, "upper"]

Dtest$width_shr <-
  Dtest$va_shr_upper -
  Dtest$va_shr_lower

## ===============================
## 17. VA: SHRINKAGE, CS = 50
## ===============================

va_shr_small <- t(
  vapply(
    Dtest$pred_p_shr,
    function(s) {
      va_predict(
        scores_cal = Dcal_small$pred_p_shr,
        y_cal = Dcal_small$y,
        score_new = s
      )
    },
    numeric(2)
  )
)

Dtest$va_shr_small_lower <-
  va_shr_small[, "lower"]

Dtest$va_shr_small_upper <-
  va_shr_small[, "upper"]

Dtest$width_shr_small <-
  Dtest$va_shr_small_upper -
  Dtest$va_shr_small_lower

## ===============================
## 18. VA MONOTONE-INVARIANCE CHECK
## ===============================

# Monotone recalibration preserves score ordering.
# With the same calibration sample, the VA widths
# should therefore coincide up to numerical precision.

max_width_difference <- max(
  abs(
    Dtest$width_over -
      Dtest$width_shr
  )
)

cat(
  "\nMaximum VA width difference ",
  "(overconfident vs shrinkage-corrected, CS=100): ",
  signif(max_width_difference, 8),
  "\n",
  sep = ""
)

## ===============================
## 19. VA WIDTH DATA
## ===============================

width_df <- bind_rows(
  data.frame(
    width = Dtest$width_over,
    Scenario = "Overconfident\n(CS = 100)"
  ),
  data.frame(
    width = Dtest$width_shr,
    Scenario = "Shrinkage-corrected\n(CS = 100)"
  ),
  data.frame(
    width = Dtest$width_shr_small,
    Scenario = "Shrinkage-corrected\n(limited CS = 50)"
  )
)

width_df$Scenario <- factor(
  width_df$Scenario,
  levels = c(
    "Overconfident\n(CS = 100)",
    "Shrinkage-corrected\n(CS = 100)",
    "Shrinkage-corrected\n(limited CS = 50)"
  )
)

## ===============================
## 20. VA WIDTH SUMMARY
## ===============================

width_summary <- width_df |>
  group_by(Scenario) |>
  summarise(
    n = n(),
    mean_width = mean(width),
    median_width = median(width),
    Q1_width = quantile(width, 0.25),
    Q3_width = quantile(width, 0.75),
    IQR_width = IQR(width),
    q90_width = quantile(width, 0.90),
    q95_width = quantile(width, 0.95),
    max_width = max(width),
    prop_width_lt_001 = mean(width < 0.01),
    prop_width_lt_005 = mean(width < 0.05),
    prop_degenerate = mean(width < 1e-12),
    .groups = "drop"
  )

print(width_summary)

## ===============================
## 21. VA WIDTH DENSITIES
## ===============================

p_width_density <- ggplot(
  width_df,
  aes(
    x = width,
    fill = Scenario
  )
) +
  geom_density(
    alpha = 0.55,
    adjust = 0.8
  ) +
  facet_wrap(
    ~ Scenario,
    nrow = 1
  ) +
  scale_fill_brewer(
    palette = "Set3"
  ) +
  labs(
    x = "Venn–Abers interval width",
    y = "Density"
  ) +
  theme_bw(base_size = 13) +
  theme(
    legend.position = "none"
  )

print(p_width_density)

## ===============================
## 22. VA WIDTH HISTOGRAMS
## ===============================

p_width_hist <- ggplot(
  width_df,
  aes(
    x = width,
    fill = Scenario
  )
) +
  geom_histogram(
    bins = 25,
    color = "black"
  ) +
  facet_wrap(
    ~ Scenario,
    nrow = 1
  ) +
  scale_fill_brewer(
    palette = "Set3"
  ) +
  labs(
    x = "Venn–Abers interval width",
    y = "Count"
  ) +
  theme_bw(base_size = 13) +
  theme(
    legend.position = "none"
  )

print(p_width_hist)

## ===============================
## 23. DIRECT DENSITY COMPARISON:
##     OVERCONFIDENT VS SHRINKAGE
## ===============================

p_width_overlay <- width_df |>
  filter(
    Scenario %in% c(
      "Overconfident\n(CS = 100)",
      "Shrinkage-corrected\n(CS = 100)"
    )
  ) |>
  ggplot(
    aes(
      x = width,
      fill = Scenario
    )
  ) +
  geom_density(
    alpha = 0.45
  ) +
  scale_fill_brewer(
    palette = "Set3"
  ) +
  labs(
    x = "Venn–Abers interval width",
    y = "Density",
    fill = ""
  ) +
  theme_bw(base_size = 13) +
  theme(
    legend.position = "bottom"
  )

print(p_width_overlay)

## ===============================
## 24. ISOTONIC CALIBRATION MAPPING
## ===============================

fit_isotonic <- function(scores, y) {
  
  ord <- order(
    scores,
    seq_along(scores)
  )
  
  iso <- isoreg(
    scores[ord],
    y[ord]
  )
  
  data.frame(
    score = scores[ord],
    calibrated_p = iso$yf
  )
}

iso_over <- fit_isotonic(
  Dcal$pred_p_over,
  Dcal$y
) |>
  mutate(
    Scenario =
      "Overconfident\n(CS = 100)"
  )

iso_shr <- fit_isotonic(
  Dcal$pred_p_shr,
  Dcal$y
) |>
  mutate(
    Scenario =
      "Shrinkage-corrected\n(CS = 100)"
  )

iso_shr_small <- fit_isotonic(
  Dcal_small$pred_p_shr,
  Dcal_small$y
) |>
  mutate(
    Scenario =
      "Shrinkage-corrected\n(limited CS = 50)"
  )

iso_df <- bind_rows(
  iso_over,
  iso_shr,
  iso_shr_small
)

iso_df$Scenario <- factor(
  iso_df$Scenario,
  levels = c(
    "Overconfident\n(CS = 100)",
    "Shrinkage-corrected\n(CS = 100)",
    "Shrinkage-corrected\n(limited CS = 50)"
  )
)

## ===============================
## 25. ISOTONIC STEP-FUNCTION PLOT
## ===============================

p_iso <- ggplot(
  iso_df,
  aes(
    x = score,
    y = calibrated_p
  )
) +
  geom_step(
    linewidth = 1.1,
    color = "black"
  ) +
  facet_wrap(
    ~ Scenario,
    nrow = 1
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25)
  ) +
  labs(
    x = "Predicted probability",
    y = "Calibrated probability"
  ) +
  theme_bw(base_size = 13)

print(p_iso)

## ===============================
## 26. ISOTONIC RESOLUTION SUMMARY
## ===============================

isotonic_summary <- iso_df |>
  group_by(Scenario) |>
  summarise(
    n_calibration_points = n(),
    n_isotonic_levels =
      n_distinct(
        round(
          calibrated_p,
          12
        )
      ),
    mean_points_per_level =
      n_calibration_points /
      n_isotonic_levels,
    .groups = "drop"
  )

print(isotonic_summary)
