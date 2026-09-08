################################################################################
## FREQUENTIST AND BAYESIAN LOGISTIC REGRESSION
## Individual prediction uncertainty and decision fragility
################################################################################

## ===============================
## 0. SETUP
## ===============================
library(haven)
library(dplyr)
library(purrr)
library(tibble)
library(tidyr)
library(ggplot2)
library(writexl)
library(rstanarm)

set.seed(123)

## ===============================
## 1. LOAD AND PREPROCESS DATA
## ===============================
df <- read_dta("Truffle_ready.dta") |>
  as.data.frame() |>
  na.omit()

df$Adverse_outcome <- as.numeric(as.character(df$Adverse_outcome))
df$Caucasian <- factor(df$Caucasian)
df$Diabetes_cat <- factor(df$Diabetes_cat)
df$Smoking <- factor(df$Smoking)

if (!all(df$Adverse_outcome %in% c(0, 1))) {
  stop("Adverse_outcome must be coded as 0/1.")
}

y <- df$Adverse_outcome
n <- nrow(df)

cat("N =", n, "\n")
cat("Events =", sum(y), "\n")
cat("Prevalence =", round(mean(y), 4), "\n")

## ===============================
## 2. ANALYSIS SETTINGS
## ===============================
B_boot <- 1000

thresholds_to_test <- c(
  0.05,
  0.08,
  0.10,
  0.15
)

## ===============================
## 3. HELPER FUNCTIONS
## ===============================
clamp01 <- function(p) {
  pmin(
    pmax(p, 1e-12),
    1 - 1e-12
  )
}

safe_q <- function(x, prob) {
  if (all(is.na(x))) {
    return(NA_real_)
  }
  
  as.numeric(
    quantile(
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

## ===============================
## 4. FREQUENTIST LOGISTIC REGRESSION
## ===============================
freq_model <- glm(
  Adverse_outcome ~ .,
  data = df,
  family = binomial(link = "logit")
)

summary(freq_model)

pred_freq <- clamp01(
  predict(
    freq_model,
    newdata = df,
    type = "response"
  )
)

## ===============================
## 5. BOOTSTRAP PREDICTION DISTRIBUTIONS
## ===============================
bootstrap_prediction_matrix <- function(data, B = 1000) {
  
  n <- nrow(data)
  
  pred_boot <- matrix(
    NA_real_,
    nrow = n,
    ncol = B
  )
  
  valid <- logical(B)
  pb <- make_progress(B)
  
  on.exit(
    try(close(pb), silent = TRUE),
    add = TRUE
  )
  
  for (b in seq_len(B)) {
    
    utils::setTxtProgressBar(pb, b)
    
    # Ordinary non-stratified bootstrap
    idx_b <- sample(
      seq_len(n),
      size = n,
      replace = TRUE
    )
    
    data_b <- data[
      idx_b,
      ,
      drop = FALSE
    ]
    
    model_b <- try(
      glm(
        Adverse_outcome ~ .,
        data = data_b,
        family = binomial(link = "logit")
      ),
      silent = TRUE
    )
    
    if (inherits(model_b, "try-error")) {
      next
    }
    
    pred_b <- try(
      predict(
        model_b,
        newdata = data,
        type = "response"
      ),
      silent = TRUE
    )
    
    if (inherits(pred_b, "try-error")) {
      next
    }
    
    if (any(!is.finite(pred_b))) {
      next
    }
    
    pred_boot[, b] <- clamp01(pred_b)
    valid[b] <- TRUE
  }
  
  cat("\n")
  
  pred_boot <- pred_boot[
    ,
    valid,
    drop = FALSE
  ]
  
  cat(
    "Valid bootstrap replications:",
    ncol(pred_boot),
    "of",
    B,
    "\n"
  )
  
  if (ncol(pred_boot) < 500) {
    warning(
      "Few valid bootstrap replicates. Check separation/convergence."
    )
  }
  
  pred_boot
}

pred_boot <- bootstrap_prediction_matrix(
  data = df,
  B = B_boot
)

## ===============================
## 6. FREQUENTIST INDIVIDUAL UNCERTAINTY
## ===============================
patient_freq <- tibble(
  patient_id = seq_len(n),
  outcome = y,
  pred_freq = pred_freq,
  
  boot_mean = rowMeans(
    pred_boot,
    na.rm = TRUE
  ),
  
  boot_median = apply(
    pred_boot,
    1,
    median,
    na.rm = TRUE
  ),
  
  boot_lower = apply(
    pred_boot,
    1,
    safe_q,
    prob = 0.025
  ),
  
  boot_upper = apply(
    pred_boot,
    1,
    safe_q,
    prob = 0.975
  )
) |>
  mutate(
    boot_width = boot_upper - boot_lower
  )

## ===============================
## 7. BAYESIAN LOGISTIC REGRESSION
## ===============================
set.seed(123)

bayes_model <- stan_glm(
  Adverse_outcome ~ .,
  data = df,
  family = binomial(link = "logit"),
  
  # Weakly informative priors
  prior = normal(
    location = 0,
    scale = 2.5,
    autoscale = TRUE
  ),
  
  prior_intercept = normal(
    location = 0,
    scale = 5,
    autoscale = FALSE
  ),
  
  chains = 4,
  iter = 4000,
  warmup = 2000,
  seed = 123,
  adapt_delta = 0.95,
  refresh = 100
)

print(bayes_model)
prior_summary(bayes_model)

## ===============================
## 8. BAYESIAN MODEL SUMMARY
## ===============================
bayes_summary <- summary(bayes_model)
print(bayes_summary)

## ===============================
## 9. POSTERIOR PREDICTED RISKS
## ===============================
# Rows = posterior draws
# Columns = patients
post_prob <- posterior_epred(
  bayes_model,
  newdata = df
)

dim(post_prob)

## ===============================
## 10. BAYESIAN INDIVIDUAL UNCERTAINTY
## ===============================
patient_bayes <- tibble(
  patient_id = seq_len(n),
  outcome = y,
  
  bayes_mean = colMeans(
    post_prob
  ),
  
  bayes_median = apply(
    post_prob,
    2,
    median
  ),
  
  bayes_lower = apply(
    post_prob,
    2,
    quantile,
    probs = 0.025
  ),
  
  bayes_upper = apply(
    post_prob,
    2,
    quantile,
    probs = 0.975
  )
) |>
  mutate(
    bayes_width = bayes_upper - bayes_lower
  )

## ===============================
## 11. MERGE PATIENT-LEVEL RESULTS
## ===============================
patient_compare <- patient_freq |>
  left_join(
    patient_bayes,
    by = c(
      "patient_id",
      "outcome"
    )
  )

## ===============================
## 12. OVERALL UNCERTAINTY SUMMARY
## ===============================
uncertainty_summary <- tibble(
  Method = c(
    "Bootstrap",
    "Bayesian"
  ),
  
  Mean_interval_width = c(
    mean(
      patient_compare$boot_width,
      na.rm = TRUE
    ),
    mean(
      patient_compare$bayes_width,
      na.rm = TRUE
    )
  ),
  
  Median_interval_width = c(
    median(
      patient_compare$boot_width,
      na.rm = TRUE
    ),
    median(
      patient_compare$bayes_width,
      na.rm = TRUE
    )
  ),
  
  Q25_interval_width = c(
    quantile(
      patient_compare$boot_width,
      0.25,
      na.rm = TRUE
    ),
    quantile(
      patient_compare$bayes_width,
      0.25,
      na.rm = TRUE
    )
  ),
  
  Q75_interval_width = c(
    quantile(
      patient_compare$boot_width,
      0.75,
      na.rm = TRUE
    ),
    quantile(
      patient_compare$bayes_width,
      0.75,
      na.rm = TRUE
    )
  )
)

print(uncertainty_summary)

## ===============================
## 13. DECISION FRAGILITY: BOOTSTRAP
## ===============================
fragility_boot <- map_dfr(
  thresholds_to_test,
  function(thr) {
    
    fragile <-
      patient_compare$boot_lower <= thr &
      patient_compare$boot_upper >= thr
    
    tibble(
      threshold = thr,
      Method = "Bootstrap",
      n_fragile = sum(
        fragile,
        na.rm = TRUE
      ),
      n_total = sum(
        !is.na(fragile)
      ),
      prop_fragile = mean(
        fragile,
        na.rm = TRUE
      )
    )
  }
)

## ===============================
## 14. DECISION FRAGILITY: BAYESIAN
## ===============================
fragility_bayes <- map_dfr(
  thresholds_to_test,
  function(thr) {
    
    fragile <-
      patient_compare$bayes_lower <= thr &
      patient_compare$bayes_upper >= thr
    
    tibble(
      threshold = thr,
      Method = "Bayesian",
      n_fragile = sum(
        fragile,
        na.rm = TRUE
      ),
      n_total = sum(
        !is.na(fragile)
      ),
      prop_fragile = mean(
        fragile,
        na.rm = TRUE
      )
    )
  }
)

## ===============================
## 15. FINAL FRAGILITY TABLE
## ===============================
fragility_long <- bind_rows(
  fragility_boot,
  fragility_bayes
)

fragility_table <- fragility_long |>
  mutate(
    Fragility_percent =
      100 * prop_fragile
  ) |>
  select(
    threshold,
    Method,
    n_fragile,
    n_total,
    Fragility_percent
  ) |>
  pivot_wider(
    names_from = Method,
    values_from = c(
      n_fragile,
      Fragility_percent
    )
  ) |>
  arrange(threshold)

print(fragility_table)

## ===============================
## 16. PATIENT-LEVEL FRAGILITY AGREEMENT
## ===============================
fragility_agreement <- map_dfr(
  thresholds_to_test,
  function(thr) {
    
    boot_fragile <-
      patient_compare$boot_lower <= thr &
      patient_compare$boot_upper >= thr
    
    bayes_fragile <-
      patient_compare$bayes_lower <= thr &
      patient_compare$bayes_upper >= thr
    
    tibble(
      threshold = thr,
      
      both_fragile = sum(
        boot_fragile &
          bayes_fragile,
        na.rm = TRUE
      ),
      
      bootstrap_only = sum(
        boot_fragile &
          !bayes_fragile,
        na.rm = TRUE
      ),
      
      bayesian_only = sum(
        !boot_fragile &
          bayes_fragile,
        na.rm = TRUE
      ),
      
      neither = sum(
        !boot_fragile &
          !bayes_fragile,
        na.rm = TRUE
      ),
      
      agreement_percent =
        100 *
        mean(
          boot_fragile == bayes_fragile,
          na.rm = TRUE
        )
    )
  }
)

print(fragility_agreement)

## ===============================
## 17. SELECT PATIENTS FOR FIGURE
## ===============================
# Patients are selected across the predicted-risk distribution
# rather than according to agreement or disagreement between methods.

n_patients_plot <- 20

risk_quantiles <- seq(
  0.02,
  0.98,
  length.out = n_patients_plot
)

selected_ids <- unique(
  sapply(
    risk_quantiles,
    function(q) {
      
      target <- quantile(
        patient_compare$pred_freq,
        q,
        na.rm = TRUE
      )
      
      which.min(
        abs(
          patient_compare$pred_freq -
            target
        )
      )
    }
  )
)

plot_patients <- patient_compare |>
  filter(
    patient_id %in% selected_ids
  ) |>
  arrange(pred_freq) |>
  mutate(
    display_order = row_number(),
    patient_label = paste0(
      "Patient ",
      display_order
    )
  )

## ===============================
## 18. PREPARE INTERVAL-PLOT DATA
## ===============================
plot_intervals <- bind_rows(
  
  plot_patients |>
    transmute(
      patient_id,
      patient_label,
      display_order,
      Method = "Bootstrap",
      Estimate = pred_freq,
      Lower = boot_lower,
      Upper = boot_upper
    ),
  
  plot_patients |>
    transmute(
      patient_id,
      patient_label,
      display_order,
      Method = "Bayesian",
      Estimate = bayes_mean,
      Lower = bayes_lower,
      Upper = bayes_upper
    )
  
) |>
  mutate(
    patient_label = factor(
      patient_label,
      levels = plot_patients$patient_label
    )
  )

## ===============================
## 19. INDIVIDUAL UNCERTAINTY FIGURE
## ===============================
p_intervals <- ggplot(
  plot_intervals,
  aes(
    x = Estimate,
    y = patient_label,
    shape = Method
  )
) +
  geom_errorbar(
    aes(
      xmin = Lower,
      xmax = Upper
    ),
    orientation = "y",
    position = position_dodge(width = 0.5),
    width = 0
  ) +
  geom_point(
    position = position_dodge(width = 0.5),
    size = 2.5
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    axis.title.y = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  labs(
    x = "Predicted probability",
    shape = ""
  )

print(p_intervals)

## ===============================
## 20. FIGURE WITH DECISION THRESHOLD
## ===============================
threshold_plot <- 0.10

p_intervals_threshold <- p_intervals +
  geom_vline(
    xintercept = threshold_plot,
    linetype = 2
  ) +
  labs(
    subtitle = paste0(
      "Dashed line: decision threshold = ",
      threshold_plot
    )
  )

print(p_intervals_threshold)

