############################################
## SIMULATION OF UNDERFITTING
## CAUSED BY EXCESSIVE SHRINKAGE
############################################

## ===============================
## 0. SETUP
## ===============================
library(ggplot2)
library(dplyr)
library(mgcv)
library(tidyr)

set.seed(101)

n <- 3000
n_boot <- 50

## ===============================
## 1. SIMULATE DATA
## ===============================

# Generate predictor and true event probabilities
x <- rnorm(n)
logit_risk <- -1 + 2.5 * x
prob_risk <- plogis(logit_risk)

# Generate binary outcomes
y <- rbinom(n, 1, prob_risk)

df <- data.frame(y, x)

## ===============================
## 2. FIT CORRECT AND UNDERFITTED MODELS
## ===============================

# Correctly specified logistic regression model
mod_correct <- glm(
  y ~ x,
  data = df,
  family = binomial
)

# Mimic underfitting by excessive coefficient shrinkage
beta_shrunk <- coef(mod_correct) * 0.7

df$p_under <- plogis(
  beta_shrunk[1] + beta_shrunk[2] * df$x
)

df$p_correct <- predict(
  mod_correct,
  type = "response"
)

## ===============================
## 3. CALIBRATION CURVE FUNCTION
## ===============================

get_cal_curve <- function(data, outcome_col, pred_col, k = 4) {
  
  form <- as.formula(
    paste(
      outcome_col,
      "~ s(",
      pred_col,
      ", k =",
      k,
      ")"
    )
  )
  
  m <- gam(
    form,
    data = data,
    family = binomial
  )
  
  grid <- data.frame(
    p = seq(
      min(data[[pred_col]]),
      max(data[[pred_col]]),
      length.out = 100
    )
  )
  
  colnames(grid) <- pred_col
  
  grid$y <- as.numeric(
    predict(
      m,
      grid,
      type = "response"
    )
  )
  
  grid
}

## ===============================
## 4. BOOTSTRAP CALIBRATION CURVES
## ===============================

boot_curves_correct <- list()
boot_curves_under <- list()

for (i in seq_len(n_boot)) {
  
  df_boot <- df[
    sample(
      nrow(df),
      replace = TRUE
    ),
  ]
  
  # Correct model
  c_corr <- get_cal_curve(
    df_boot,
    "y",
    "p_correct"
  )
  
  boot_curves_correct[[i]] <- data.frame(
    x = c_corr$p_correct,
    y = c_corr$y,
    id = i
  )
  
  # Underfitted model
  c_under <- get_cal_curve(
    df_boot,
    "y",
    "p_under"
  )
  
  boot_curves_under[[i]] <- data.frame(
    x = c_under$p_under,
    y = c_under$y,
    id = i
  )
}

df_boot_corr <- bind_rows(
  boot_curves_correct
)

df_boot_under <- bind_rows(
  boot_curves_under
)

## ===============================
## 5. CALIBRATION CURVES IN FULL SAMPLE
## ===============================

main_curve_correct <- get_cal_curve(
  df,
  "y",
  "p_correct"
)

main_curve_under <- get_cal_curve(
  df,
  "y",
  "p_under"
)

## ===============================
## 6. CALIBRATION PLOT
## ===============================

ggplot() +
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
    data = df_boot_corr,
    aes(
      x = x,
      y = y,
      group = id
    ),
    color = "green3",
    alpha = 0.1
  ) +
  geom_line(
    data = df_boot_under,
    aes(
      x = x,
      y = y,
      group = id
    ),
    color = "blue",
    alpha = 0.1
  ) +
  geom_line(
    data = main_curve_under,
    aes(
      x = p_under,
      y = y,
      color = "Underfitted (over-penalized)"
    ),
    linewidth = 1.5
  ) +
  geom_line(
    data = main_curve_correct,
    aes(
      x = p_correct,
      y = y,
      color = "Correctly specified"
    ),
    linewidth = 1.5
  ) +
  scale_color_manual(
    values = c(
      "Perfect calibration" = "red",
      "Underfitted (over-penalized)" = "blue",
      "Correctly specified" = "green3"
    )
  ) +
  labs(
    x = "Predicted probability",
    y = "Observed probability",
    color = ""
  ) +
  theme_minimal() +
  coord_cartesian(
    xlim = c(0, 1),
    ylim = c(0, 1)
  ) +
  theme(
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11),
    legend.position = "bottom",
    legend.text = element_text(size = 10),
    panel.grid.minor = element_blank()
  )