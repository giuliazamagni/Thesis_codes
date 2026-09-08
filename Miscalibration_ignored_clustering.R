############################################
## SIMULATION OF MISCALIBRATION
## CAUSED BY IGNORED CLUSTERING
############################################

## ===============================
## 0. SETUP
## ===============================
library(tidyverse)
library(lme4)
library(ggpubr)

set.seed(123)

n_centers <- 6
n_per_center <- 500
n_total <- n_centers * n_per_center
B <- 80

## ===============================
## 1. SIMULATE CLUSTERED DATA
## ===============================

# Center-specific random intercepts
center_effects <- rnorm(
  n_centers,
  mean = 0,
  sd = 1.5
)

data <- data.frame(
  center = factor(
    rep(seq_len(n_centers), each = n_per_center)
  ),
  x = rnorm(n_total)
)

# True data-generating mechanism includes center-level heterogeneity
data$logit <- -1 +
  1.2 * data$x +
  center_effects[as.numeric(data$center)]

data$prob_true <- plogis(data$logit)
data$y <- rbinom(n_total, 1, data$prob_true)

## ===============================
## 2. CALIBRATION CURVE FUNCTION
## ===============================

get_curve <- function(pred, y) {
  as.data.frame(
    lowess(
      pred,
      y,
      iter = 0
    )
  )
}

## ===============================
## 3. FIT SINGLE-LEVEL AND MULTILEVEL MODELS
## ===============================

# Single-level logistic regression ignoring clustering
mod_single <- glm(
  y ~ x,
  data = data,
  family = binomial
)

data$pred_single <- predict(
  mod_single,
  type = "response"
)

# Multilevel logistic regression with center-specific random intercept
mod_multi <- glmer(
  y ~ x + (1 | center),
  data = data,
  family = binomial
)

data$pred_multi <- predict(
  mod_multi,
  type = "response"
)

## ===============================
## 4. APPARENT CALIBRATION CURVES
## ===============================

# Center-specific calibration curves
curve_single_centers <- data |>
  group_by(center) |>
  group_modify(
    ~ get_curve(
      .x$pred_single,
      .x$y
    )
  )

curve_multi_centers <- data |>
  group_by(center) |>
  group_modify(
    ~ get_curve(
      .x$pred_multi,
      .x$y
    )
  )

# Overall calibration curves
curve_single_global <- get_curve(
  data$pred_single,
  data$y
)

curve_multi_global <- get_curve(
  data$pred_multi,
  data$y
)

## ===============================
## 5. BOOTSTRAP CALIBRATION CURVES
## ===============================

# Bootstrap observations within centers.
# This is intentionally an apparent, not out-of-bag, assessment.

boot_single <- map_df(
  seq_len(B),
  function(b) {
    
    d <- data |>
      group_by(center) |>
      sample_frac(replace = TRUE) |>
      ungroup()
    
    m <- glm(
      y ~ x,
      data = d,
      family = binomial
    )
    
    d$pred <- predict(
      m,
      type = "response"
    )
    
    d |>
      group_by(center) |>
      group_modify(
        ~ get_curve(
          .x$pred,
          .x$y
        )
      ) |>
      mutate(run = b)
  }
)

boot_multi <- map_df(
  seq_len(B),
  function(b) {
    
    d <- data |>
      group_by(center) |>
      sample_frac(replace = TRUE) |>
      ungroup()
    
    m <- glmer(
      y ~ x + (1 | center),
      data = d,
      family = binomial
    )
    
    d$pred <- predict(
      m,
      type = "response",
      allow.new.levels = TRUE
    )
    
    d |>
      group_by(center) |>
      group_modify(
        ~ get_curve(
          .x$pred,
          .x$y
        )
      ) |>
      mutate(run = b)
  }
)

## ===============================
## 6. SINGLE-LEVEL MODEL PLOT
## ===============================

plot_single <- ggplot() +
  annotate(
    "segment",
    x = 0,
    y = 0,
    xend = 1,
    yend = 1,
    colour = "red",
    linewidth = 1
  ) +
  geom_line(
    data = boot_single,
    aes(
      x,
      y,
      group = interaction(run, center),
      colour = center
    ),
    alpha = 0.12,
    linewidth = 0.4
  ) +
  geom_line(
    data = curve_single_centers,
    aes(
      x,
      y,
      colour = center
    ),
    linewidth = 1.4
  ) +
  geom_line(
    data = curve_single_global,
    aes(x, y),
    colour = "black",
    linewidth = 1.6
  ) +
  scale_color_brewer(
    palette = "Set2",
    name = "Centers"
  ) +
  coord_cartesian(
    xlim = c(0, 1),
    ylim = c(0, 1)
  ) +
  labs(
    title = "a. Single-level model",
    x = "Predicted probability",
    y = "Observed probability"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom"
  )

## ===============================
## 7. MULTILEVEL MODEL PLOT
## ===============================

plot_multi <- ggplot() +
  annotate(
    "segment",
    x = 0,
    y = 0,
    xend = 1,
    yend = 1,
    colour = "red",
    linewidth = 1
  ) +
  geom_line(
    data = boot_multi,
    aes(
      x,
      y,
      group = interaction(run, center),
      colour = center
    ),
    alpha = 0.12,
    linewidth = 0.4
  ) +
  geom_line(
    data = curve_multi_centers,
    aes(
      x,
      y,
      colour = center
    ),
    linewidth = 1.4
  ) +
  geom_line(
    data = curve_multi_global,
    aes(x, y),
    colour = "darkgreen",
    linewidth = 1.6
  ) +
  scale_color_brewer(
    palette = "Set2",
    name = "Centers"
  ) +
  coord_cartesian(
    xlim = c(0, 1),
    ylim = c(0, 1)
  ) +
  labs(
    title = "b. Multilevel model",
    x = "Predicted probability",
    y = "Observed probability"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom"
  )

## ===============================
## 8. COMBINE AND EXPORT FIGURE
## ===============================

p <- ggarrange(
  plot_single,
  plot_multi,
  ncol = 1,
  nrow = 2,
  common.legend = TRUE,
  legend = "bottom"
)

ggsave(
  filename = "calibration_multilevel.tiff",
  plot = p,
  device = "tiff",
  bg = "white",
  width = 8,
  height = 10,
  dpi = 600,
  compression = "lzw"
)