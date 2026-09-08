############################################
## SIMULATION OF OVERCONFIDENCE AND SHRINKAGE
## WITH OUT-OF-BAG CALIBRATION ASSESSMENT
############################################

## ===============================
## 0. SETUP
## ===============================
library(ggplot2)
library(dplyr)
library(mgcv)

set.seed(123)

n <- 200
n_boot <- 100

## ===============================
## 1. SIMULATE DATA
## ===============================

# True individual event probabilities
true_p <- runif(n, 0.05, 0.95)

# Binary outcomes generated from the true probabilities
y <- rbinom(n, 1, true_p)

# Generate overconfident predictions by exaggerating the true log-odds
z <- qlogis(true_p)
pred_p_overconfident <- plogis(1.8 * z + 0.5)

df <- data.frame(
  y = y,
  pred_p = pred_p_overconfident
)

## ===============================
## 2. ESTIMATE SHRINKAGE
## ===============================

# Estimate calibration intercept and slope
fit_cal <- glm(
  y ~ qlogis(pred_p),
  data = df,
  family = binomial
)

intercept <- coef(fit_cal)[1]
s_factor <- coef(fit_cal)[2]

# Apply shrinkage on the log-odds scale
df$corrected_p <- plogis(
  intercept + s_factor * qlogis(df$pred_p)
)

## ===============================
## 3. OUT-OF-BAG BOOTSTRAP
## ===============================

boot_results <- data.frame()

for (i in seq_len(n_boot)) {
  
  idx <- sample(seq_len(n), n, replace = TRUE)
  
  # Observations not selected in the bootstrap sample
  test <- df[-unique(idx), ]
  
  # Flexible calibration curves evaluated in out-of-bag observations
  m_over <- gam(
    y ~ s(pred_p, k = 3),
    data = test,
    family = binomial
  )
  
  m_shrunk <- gam(
    y ~ s(corrected_p, k = 3),
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
          data.frame(pred_p = grid),
          type = "response"
        )
      ),
      y_shrunk = as.numeric(
        predict(
          m_shrunk,
          data.frame(corrected_p = grid),
          type = "response"
        )
      ),
      id = i
    )
  )
}

## ===============================
## 4. CALIBRATION PLOT
## ===============================

ggplot(boot_results) +
  geom_segment(
    aes(
      x = 0,
      y = 0,
      xend = 1,
      yend = 1,
      color = "Perfect calibration"
    ),
    linewidth = 1.5
  ) +
  geom_line(
    aes(x = x, y = y_over, group = id),
    color = "blue",
    alpha = 0.05,
    linewidth = 0.5
  ) +
  geom_line(
    aes(x = x, y = y_shrunk, group = id),
    color = "green3",
    alpha = 0.05,
    linewidth = 0.5
  ) +
  geom_smooth(
    aes(
      x = x,
      y = y_over,
      color = "Overconfident"
    ),
    method = "loess",
    se = FALSE,
    linewidth = 1.5
  ) +
  geom_smooth(
    aes(
      x = x,
      y = y_shrunk,
      color = "Corrected (shrunk)"
    ),
    method = "loess",
    se = FALSE,
    linewidth = 1.5
  ) +
  scale_color_manual(
    values = c(
      "Overconfident" = "blue",
      "Corrected (shrunk)" = "green3",
      "Perfect calibration" = "red"
    )
  ) +
  theme_minimal() +
  labs(
    x = "Predicted probability",
    y = "Observed probability",
    color = ""
  ) +
  theme(
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11),
    legend.position = "bottom",
    legend.text = element_text(size = 10),
    panel.grid.minor = element_blank()
  )