############################################
## SIMULATION OF MODEL MISSPECIFICATION
## DUE TO AN OMITTED INTERACTION TERM
############################################

## ===============================
## 0. SETUP
## ===============================
library(ggplot2)
library(dplyr)
library(tidyr)

set.seed(1234)

n <- 3000
n_boot <- 50

## ===============================
## 1. SIMULATE DATA
## ===============================

# Simulate biomarker and treatment exposure
biomarker_X1 <- rnorm(n, mean = 0.5, sd = 1)
drug_X2 <- rbinom(n, 1, 0.4)

# True data-generating mechanism includes an interaction
logit_toxicity <- -2.0 +
  0.5 * biomarker_X1 +
  1.0 * drug_X2 +
  2.5 * (biomarker_X1 * drug_X2)

prob_toxicity <- plogis(logit_toxicity)
toxicity_Y <- rbinom(n, 1, prob_toxicity)

df_pharma <- data.frame(
  toxicity_Y,
  biomarker_X1,
  drug_X2
)

## ===============================
## 2. BOOTSTRAP CALIBRATION CURVES
## ===============================

boot_df <- data.frame()

for (i in seq_len(n_boot)) {
  
  df_b <- df_pharma[
    sample(seq_len(n), n, replace = TRUE),
  ]
  
  # Correctly specified model
  m_corr_b <- glm(
    toxicity_Y ~ biomarker_X1 * drug_X2,
    data = df_b,
    family = binomial
  )
  
  # Misspecified model omitting the interaction term
  m_miss_b <- glm(
    toxicity_Y ~ biomarker_X1 + drug_X2,
    data = df_b,
    family = binomial
  )
  
  l_corr_b <- lowess(
    predict(m_corr_b, type = "response"),
    df_b$toxicity_Y,
    iter = 0,
    f = 0.75
  )
  
  l_miss_b <- lowess(
    predict(m_miss_b, type = "response"),
    df_b$toxicity_Y,
    iter = 0,
    f = 0.75
  )
  
  boot_df <- rbind(
    boot_df,
    data.frame(
      x = l_corr_b$x,
      y = l_corr_b$y,
      type = "Correctly specified",
      id = i
    ),
    data.frame(
      x = l_miss_b$x,
      y = l_miss_b$y,
      type = "Misspecified",
      id = i
    )
  )
}

## ===============================
## 3. FIT MODELS IN FULL SAMPLE
## ===============================

m_corr_app <- glm(
  toxicity_Y ~ biomarker_X1 * drug_X2,
  data = df_pharma,
  family = binomial
)

m_miss_app <- glm(
  toxicity_Y ~ biomarker_X1 + drug_X2,
  data = df_pharma,
  family = binomial
)

df_pharma$p_corr_app <- predict(
  m_corr_app,
  type = "response"
)

df_pharma$p_miss_app <- predict(
  m_miss_app,
  type = "response"
)

## ===============================
## 4. CALIBRATION PLOT
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
  
  # Bootstrap calibration curves
  geom_line(
    data = boot_df |> filter(type == "Correctly specified"),
    aes(x = x, y = y, group = id),
    color = "green3",
    alpha = 0.1
  ) +
  
  geom_line(
    data = boot_df |> filter(type == "Misspecified"),
    aes(x = x, y = y, group = id),
    color = "blue",
    alpha = 0.1
  ) +
  
  # Main calibration curves
  geom_smooth(
    data = df_pharma,
    aes(
      x = p_corr_app,
      y = toxicity_Y,
      color = "Correctly specified (interaction included)"
    ),
    method = "loess",
    se = FALSE,
    linewidth = 1.5,
    span = 0.75
  ) +
  
  geom_smooth(
    data = df_pharma,
    aes(
      x = p_miss_app,
      y = toxicity_Y,
      color = "Misspecified (interaction omitted)"
    ),
    method = "loess",
    se = FALSE,
    linewidth = 1.5,
    span = 0.75
  ) +
  
  scale_color_manual(
    values = c(
      "Perfect calibration" = "red",
      "Correctly specified (interaction included)" = "green3",
      "Misspecified (interaction omitted)" = "blue"
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