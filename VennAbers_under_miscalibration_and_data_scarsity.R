################################################################################
## CONFORMAL AND VENN–ABERS UNDER MISCALIBRATION
## Effect of monotone recalibration and limited calibration data
################################################################################

## ===============================
## 0. SETUP
## ===============================
library(ggplot2)
library(dplyr)
library(tidyr)

set.seed(123)

n_dev <- 200
n <- 200
n_cal <- 100
n_cal_small <- 50
alpha <- 0.05

## ===============================
## 1. DATA-GENERATING MECHANISM
## ===============================

simulate_data <- function(n) {
  
  true_p <- runif(n, 0.05, 0.95)
  y <- rbinom(n, 1, true_p)
  
  # Artificial overconfidence on the log-odds scale
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
## 2. INDEPENDENT DEVELOPMENT SAMPLE
##    FOR LOGISTIC RECALIBRATION
## ===============================

# Recalibration is estimated in an independent sample so that
# the subsequent conformal calibration and test outcomes are
# not used to construct the prediction score.

set.seed(111)

Ddev <- simulate_data(
  n_dev
)

fit_cal <- glm(
  y ~ qlogis(pred_p_over),
  data = Ddev,
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

Ddev$pred_p_shr <- recalibrate(
  Ddev$pred_p_over
)

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
## 3. INDEPENDENT EVALUATION SAMPLE
## ===============================

set.seed(123)

df <- simulate_data(
  n
)

df$pred_p_shr <- recalibrate(
  df$pred_p_over
)

## ===============================
## 4. CALIBRATION / TEST SPLIT
## ===============================

set.seed(123)

idx_cal <- sample(
  seq_len(n),
  size = n_cal,
  replace = FALSE
)

Dcal <- df[
  idx_cal,
  ,
  drop = FALSE
]

Dtest <- df[
  -idx_cal,
  ,
  drop = FALSE
]

stopifnot(
  nrow(Dcal) == n_cal,
  nrow(Dtest) == n - n_cal
)

## ===============================
## 5. LIMITED CALIBRATION SET
## ===============================

# CS = 50 is nested within CS = 100.
# The test set remains identical in all VA scenarios.

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
## 6. NON-CONFORMITY SCORES
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
## 7. NCS SUMMARY: MEDIAN AND IQR
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

print(
  ncs_summary
)

## ===============================
## 8. NCS DENSITY PLOT
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
  theme_bw(
    base_size = 13
  ) +
  theme(
    legend.position = "bottom"
  )

print(
  p_ncs
)

## ===============================
## 9. FINITE-SAMPLE CONFORMAL QUANTILE
## ===============================

# For m calibration observations:
# k = ceiling((m + 1)(1 - alpha))
# q is the kth ordered non-conformity score.

conformal_quantile <- function(scores, alpha) {
  
  m <- length(scores)
  
  k <- ceiling(
    (m + 1) *
      (1 - alpha)
  )
  
  k <- min(
    k,
    m
  )
  
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

# Probability threshold for class inclusion
tau_over <- 1 - q_over
tau_shr <- 1 - q_shr

## ===============================
## 10. SPLIT-CONFORMAL PREDICTION SETS
## ===============================

# Class 1 is included when p >= tau.
# Class 0 is included when 1-p >= tau.

C_over_1 <- (
  Dtest$pred_p_over >= tau_over
)

C_over_0 <- (
  1 - Dtest$pred_p_over >= tau_over
)

C_shr_1 <- (
  Dtest$pred_p_shr >= tau_shr
)

C_shr_0 <- (
  1 - Dtest$pred_p_shr >= tau_shr
)

## ===============================
## 11. COVERAGE AND EFFICIENCY
## ===============================

coverage_over <- mean(
  (Dtest$y == 1 & C_over_1) |
    (Dtest$y == 0 & C_over_0)
)

coverage_shr <- mean(
  (Dtest$y == 1 & C_shr_1) |
    (Dtest$y == 0 & C_shr_0)
)

# Efficiency = singleton rate
singleton_over <- mean(
  C_over_0 + C_over_1 == 1
)

singleton_shr <- mean(
  C_shr_0 + C_shr_1 == 1
)

# Two-label prediction sets
both_over <- mean(
  C_over_0 & C_over_1
)

both_shr <- mean(
  C_shr_0 & C_shr_1
)

# Empty sets, included as a diagnostic
empty_over <- mean(
  !C_over_0 & !C_over_1
)

empty_shr <- mean(
  !C_shr_0 & !C_shr_1
)

## ===============================
## 12. CONFORMAL SUMMARY TABLE
## ===============================

conformal_summary <- data.frame(
  Model = c(
    "Overconfident",
    "Shrinkage-corrected"
  ),
  
  alpha = alpha,
  
  Calibration_n = c(
    nrow(Dcal),
    nrow(Dcal)
  ),
  
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
  
  Lower_singleton_boundary = c(
    tau_over,
    tau_shr
  ),
  
  Upper_singleton_boundary = c(
    1 - tau_over,
    1 - tau_shr
  ),
  
  Coverage = c(
    coverage_over,
    coverage_shr
  ),
  
  Efficiency_singleton_rate = c(
    singleton_over,
    singleton_shr
  ),
  
  Two_label_rate = c(
    both_over,
    both_shr
  ),
  
  Empty_set_rate = c(
    empty_over,
    empty_shr
  )
)

print(
  conformal_summary
)

## ===============================
## 13. COMBINED NCS + CONFORMAL TABLE
## ===============================

table_conformal_ncs <- conformal_summary |>
  left_join(
    ncs_summary |>
      rename(
        Model = Model
      ),
    by = "Model"
  )

print(
  table_conformal_ncs
)

## ===============================
## 14. PRINT INTERPRETABLE RULES
## ===============================

cat(
  "\nOVERCONFIDENT MODEL\n",
  "q =", round(q_over, 3), "\n",
  "tau =", round(tau_over, 3), "\n",
  "{0} if p <", round(tau_over, 3), "\n",
  "{0,1} if",
  round(tau_over, 3),
  "<= p <=",
  round(1 - tau_over, 3),
  "\n",
  "{1} if p >",
  round(1 - tau_over, 3),
  "\n\n",
  sep = ""
)

cat(
  "SHRINKAGE-CORRECTED MODEL\n",
  "q =", round(q_shr, 3), "\n",
  "tau =", round(tau_shr, 3), "\n",
  "{0} if p <", round(tau_shr, 3), "\n",
  "{0,1} if",
  round(tau_shr, 3),
  "<= p <=",
  round(1 - tau_shr, 3),
  "\n",
  "{1} if p >",
  round(1 - tau_shr, 3),
  "\n\n",
  sep = ""
)

## ===============================
## 15. VENN–ABERS FUNCTION
## ===============================

# For each test score, append the test point twice:
# once with hypothetical label 0 and once with label 1.
#
# The isotonic fitted value at the appended test observation
# is extracted directly.

va_predict <- function(
    scores_cal,
    y_cal,
    score_new
) {
  
  n_cal_local <- length(
    scores_cal
  )
  
  fit_hypothetical <- function(
    y_new
  ) {
    
    scores_aug <- c(
      scores_cal,
      score_new
    )
    
    y_aug <- c(
      y_cal,
      y_new
    )
    
    ord <- order(
      scores_aug,
      seq_along(scores_aug)
    )
    
    iso <- isoreg(
      scores_aug[ord],
      y_aug[ord]
    )
    
    test_position <- which(
      ord == n_cal_local + 1L
    )
    
    iso$yf[
      test_position
    ]
  }
  
  p0 <- fit_hypothetical(
    0
  )
  
  p1 <- fit_hypothetical(
    1
  )
  
  c(
    lower = min(p0, p1),
    upper = max(p0, p1)
  )
}

## ===============================
## 16. VA: OVERCONFIDENT, CS = 100
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
## 17. VA: SHRINKAGE, CS = 100
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
## 18. VA: SHRINKAGE, CS = 50
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
## 19. MONOTONE-INVARIANCE CHECK
## ===============================

same_ranking <- identical(
  order(Dcal$pred_p_over),
  order(Dcal$pred_p_shr)
)

max_width_difference <- max(
  abs(
    Dtest$width_over -
      Dtest$width_shr
  )
)

cat(
  "Same score ranking:",
  same_ranking,
  "\n"
)

cat(
  "Maximum VA width difference:",
  signif(
    max_width_difference,
    6
  ),
  "\n\n"
)

## ===============================
## 20. VA WIDTH DATA
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
## 21. VA WIDTH SUMMARY
## ===============================

width_summary <- width_df |>
  group_by(Scenario) |>
  summarise(
    n = n(),
    
    mean_width =
      mean(width),
    
    median_width =
      median(width),
    
    Q1_width =
      quantile(
        width,
        0.25
      ),
    
    Q3_width =
      quantile(
        width,
        0.75
      ),
    
    IQR_width =
      IQR(width),
    
    q90_width =
      quantile(
        width,
        0.90
      ),
    
    q95_width =
      quantile(
        width,
        0.95
      ),
    
    max_width =
      max(width),
    
    prop_width_lt_001 =
      mean(
        width < 0.01
      ),
    
    prop_width_lt_005 =
      mean(
        width < 0.05
      ),
    
    prop_degenerate =
      mean(
        width < 1e-12
      ),
    
    .groups = "drop"
  )

print(
  width_summary
)

## ===============================
## 22. VA WIDTH DENSITY PLOT
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
  theme_bw(
    base_size = 13
  ) +
  theme(
    legend.position = "none"
  )

print(
  p_width_density
)

## ===============================
## 23. OPTIONAL WIDTH HISTOGRAM
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
  theme_bw(
    base_size = 13
  ) +
  theme(
    legend.position = "none"
  )

print(
  p_width_hist
)

## ===============================
## 24. ISOTONIC CALIBRATION MAPPING
## ===============================

fit_isotonic <- function(
    scores,
    y
) {
  
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
    breaks = seq(
      0,
      1,
      0.25
    )
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(
      0,
      1,
      0.25
    )
  ) +
  labs(
    x = "Predicted probability",
    y = "Calibrated probability"
  ) +
  theme_bw(
    base_size = 13
  )

print(
  p_iso
)

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

print(
  isotonic_summary
)