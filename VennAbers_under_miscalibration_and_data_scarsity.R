############################################
## CONFORMAL AND VENN–ABERS UNDER
## MISCALIBRATION AND LIMITED CALIBRATION DATA
############################################

## ===============================
## 0. SETUP
## ===============================
library(ggplot2)
library(dplyr)
library(tidyr)
library(mgcv)

set.seed(123)

n <- 200
n_boot <- 100
alpha <- 0.05

## ===============================
## 1. SIMULATE DATA
## ===============================

true_p <- runif(n, 0.05, 0.95)
y <- rbinom(n, 1, true_p)

# Generate overconfident predictions
z <- qlogis(true_p)
pred_p_over <- plogis(1.8 * z + 0.5)

df <- data.frame(
  y = y,
  true_p = true_p,
  pred_p_over = pred_p_over
)

## ===============================
## 2. SHRINKAGE RECALIBRATION
## ===============================

fit_cal <- glm(
  y ~ qlogis(pred_p_over),
  data = df,
  family = binomial
)

intercept <- coef(fit_cal)[1]
slope <- coef(fit_cal)[2]

df$pred_p_shr <- plogis(
  intercept + slope * qlogis(df$pred_p_over)
)

## ===============================
## 3. BOOTSTRAP CALIBRATION CURVES
## ===============================

boot_results <- data.frame()

for (i in seq_len(n_boot)) {
  
  idx <- sample(
    seq_len(n),
    n,
    replace = TRUE
  )
  
  test <- df[-unique(idx), ]
  
  m_over <- gam(
    y ~ s(pred_p_over, k = 3),
    data = test,
    family = binomial
  )
  
  m_shr <- gam(
    y ~ s(pred_p_shr, k = 3),
    data = test,
    family = binomial
  )
  
  grid <- seq(0, 1, length.out = 100)
  
  boot_results <- rbind(
    boot_results,
    data.frame(
      x = grid,
      y_over = as.numeric(
        predict(
          m_over,
          data.frame(pred_p_over = grid),
          type = "response"
        )
      ),
      y_shrunk = as.numeric(
        predict(
          m_shr,
          data.frame(pred_p_shr = grid),
          type = "response"
        )
      ),
      id = i
    )
  )
}

## ===============================
## 4. NON-CONFORMITY SCORES
## ===============================

df$ncs_over <- 1 - ifelse(
  df$y == 1,
  df$pred_p_over,
  1 - df$pred_p_over
)

df$ncs_shr <- 1 - ifelse(
  df$y == 1,
  df$pred_p_shr,
  1 - df$pred_p_shr
)

df_ncs <- df |>
  select(ncs_over, ncs_shr) |>
  pivot_longer(
    cols = everything(),
    names_to = "model",
    values_to = "ncs"
  )

df_ncs$model <- factor(
  df_ncs$model,
  levels = c("ncs_over", "ncs_shr"),
  labels = c(
    "Overconfident",
    "Shrinkage-corrected"
  )
)

## ===============================
## 5. SPLIT CALIBRATION / TEST SET
## ===============================

set.seed(123)

idx_cal <- sample(
  seq_len(n),
  n / 2
)

Dcal <- df[idx_cal, ]
Dtest <- df[-idx_cal, ]

## ===============================
## 6. SPLIT CONFORMAL
## ===============================

ncs_over_cal <- 1 - ifelse(
  Dcal$y == 1,
  Dcal$pred_p_over,
  1 - Dcal$pred_p_over
)

ncs_shr_cal <- 1 - ifelse(
  Dcal$y == 1,
  Dcal$pred_p_shr,
  1 - Dcal$pred_p_shr
)

q_over <- quantile(
  ncs_over_cal,
  probs = 1 - alpha,
  type = 8
)

q_shr <- quantile(
  ncs_shr_cal,
  probs = 1 - alpha,
  type = 8
)

tau_over <- 1 - q_over
tau_shr <- 1 - q_shr

# Prediction sets
C_over_1 <- Dtest$pred_p_over >= tau_over
C_over_0 <- (1 - Dtest$pred_p_over) >= tau_over

C_shr_1 <- Dtest$pred_p_shr >= tau_shr
C_shr_0 <- (1 - Dtest$pred_p_shr) >= tau_shr

cover_over <- mean(
  (Dtest$y == 1 & C_over_1) |
    (Dtest$y == 0 & C_over_0)
)

cover_shr <- mean(
  (Dtest$y == 1 & C_shr_1) |
    (Dtest$y == 0 & C_shr_0)
)

singleton_over <- mean(
  (C_over_1 + C_over_0) == 1
)

singleton_shr <- mean(
  (C_shr_1 + C_shr_0) == 1
)

summary_conformal <- data.frame(
  Model = c(
    "Overconfident",
    "Shrinkage-corrected"
  ),
  Coverage = c(
    cover_over,
    cover_shr
  ),
  SingletonRate = c(
    singleton_over,
    singleton_shr
  ),
  Tau = c(
    tau_over,
    tau_shr
  )
)

print(summary_conformal)

## ===============================
## 7. NON-CONFORMITY SCORE DENSITY
## ===============================

p_ncs <- ggplot(
  df_ncs,
  aes(
    x = ncs,
    fill = model
  )
) +
  geom_density(alpha = 0.6) +
  scale_fill_brewer(
    palette = "Set3"
  ) +
  labs(
    x = "Non-conformity score",
    y = "Density",
    fill = "Model"
  ) +
  theme_minimal()

print(p_ncs)

## ===============================
## 8. VENN–ABERS FUNCTION
## ===============================

va_predict <- function(
    scores_cal,
    y_cal,
    score_new
) {
  
  # Hypothetical label = 0
  scores0 <- c(scores_cal, score_new)
  y0 <- c(y_cal, 0)
  ord0 <- order(scores0)
  
  iso0 <- isoreg(
    scores0[ord0],
    y0[ord0]
  )
  
  p0 <- approx(
    iso0$x,
    iso0$yf,
    xout = score_new,
    rule = 2
  )$y
  
  # Hypothetical label = 1
  scores1 <- c(scores_cal, score_new)
  y1 <- c(y_cal, 1)
  ord1 <- order(scores1)
  
  iso1 <- isoreg(
    scores1[ord1],
    y1[ord1]
  )
  
  p1 <- approx(
    iso1$x,
    iso1$yf,
    xout = score_new,
    rule = 2
  )$y
  
  c(
    min(p0, p1),
    max(p0, p1)
  )
}

## ===============================
## 9. VENN–ABERS: OVERCONFIDENT
## ===============================

va_over <- t(
  sapply(
    Dtest$pred_p_over,
    function(s) {
      va_predict(
        Dcal$pred_p_over,
        Dcal$y,
        s
      )
    }
  )
)

Dtest$width_over <- (
  va_over[, 2] -
    va_over[, 1]
)

## ===============================
## 10. VENN–ABERS: SHRINKAGE
## ===============================

va_shr <- t(
  sapply(
    Dtest$pred_p_shr,
    function(s) {
      va_predict(
        Dcal$pred_p_shr,
        Dcal$y,
        s
      )
    }
  )
)

Dtest$width_shr <- (
  va_shr[, 2] -
    va_shr[, 1]
)

## ===============================
## 11. LIMITED CALIBRATION SET
## ===============================

set.seed(456)

idx_cal_small <- sample(
  seq_len(n),
  50
)

Dcal_small <- df[idx_cal_small, ]
Dtest_small <- df[-idx_cal_small, ]

va_shr_small <- t(
  sapply(
    Dtest_small$pred_p_shr,
    function(s) {
      va_predict(
        Dcal_small$pred_p_shr,
        Dcal_small$y,
        s
      )
    }
  )
)

Dtest_small$width <- (
  va_shr_small[, 2] -
    va_shr_small[, 1]
)

## ===============================
## 12. VENN–ABERS INTERVAL WIDTH
## ===============================

width_df <- bind_rows(
  data.frame(
    width = Dtest$width_over,
    Scenario = "Overconfident"
  ),
  data.frame(
    width = Dtest$width_shr,
    Scenario = "Shrinkage"
  ),
  data.frame(
    width = Dtest_small$width,
    Scenario = "Shrinkage (limited CS)"
  )
)

p_width <- ggplot(
  width_df,
  aes(
    x = width,
    fill = Scenario
  )
) +
  geom_histogram(
    bins = 30,
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

print(p_width)

## ===============================
## 13. ISOTONIC CALIBRATION FUNCTION
## ===============================

fit_isotonic <- function(scores, y) {
  
  ord <- order(scores)
  
  iso <- isoreg(
    scores[ord],
    y[ord]
  )
  
  data.frame(
    score = iso$x,
    calibrated_p = iso$yf
  )
}

## ===============================
## 14. ISOTONIC MAPPINGS
## ===============================

iso_over <- fit_isotonic(
  Dcal$pred_p_over,
  Dcal$y
) |>
  mutate(
    Scenario = "Overconfident (CS = 100)"
  )

iso_shr <- fit_isotonic(
  Dcal$pred_p_shr,
  Dcal$y
) |>
  mutate(
    Scenario = "Shrinkage (CS = 100)"
  )

iso_shr_small <- fit_isotonic(
  Dcal_small$pred_p_shr,
  Dcal_small$y
) |>
  mutate(
    Scenario = "Shrinkage (limited CS = 50)"
  )

iso_df <- bind_rows(
  iso_over,
  iso_shr,
  iso_shr_small
)

iso_df$Scenario <- factor(
  iso_df$Scenario,
  levels = c(
    "Overconfident (CS = 100)",
    "Shrinkage (CS = 100)",
    "Shrinkage (limited CS = 50)"
  )
)

## ===============================
## 15. ISOTONIC STEP-FUNCTION PLOT
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
  coord_cartesian(
    xlim = c(0, 1),
    ylim = c(0, 1)
  ) +
  labs(
    x = "Predicted probability",
    y = "Calibrated probability"
  ) +
  theme_bw(base_size = 13)

print(p_iso)