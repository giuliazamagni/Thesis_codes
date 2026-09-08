################################################################################
## SURVIVAL PREDICTION AND DECISION FRAGILITY
## Cox proportional hazards vs Random Survival Forest
## Endpoint: AMI or stroke
## Prediction horizon: 36 months
################################################################################

## ===============================
## 0. SETUP
## ===============================
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

## ===============================
## 1. ANALYSIS SETTINGS
## ===============================
MODEL_ORIENTATION <- "auc_screening_CoxPH_RSF_shared_ordinary_boot500"

B <- 500
n_perm_importance <- 5

rf_tune_trees <- 750
rf_final_trees <- 1500

time_horizon <- 36

thresholds_dca <- seq(
  0.01,
  0.99,
  by = 0.01
)

thresholds_interest <- c(
  0.05,
  0.08,
  0.10,
  0.15,
  0.20,
  0.25
)

## ===============================
## 2. LOAD AND PREPROCESS DATA
## ===============================
df <- read_dta("INCLISAN_ready.dta") |>
  as.data.frame() |>
  select(
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
  na.omit() |>
  as.data.frame()

df$ANA_SESSO <- factor(df$ANA_SESSO)
df$diab_index <- factor(df$diab_index)
df$FR_FUMO <- factor(df$FR_FUMO)

stopifnot(
  all(df$death_IMA_Stroke %in% c(0, 1))
)

stopifnot(
  all(df$fup_D_IMA_STROKE >= 0)
)

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
cat(
  "Prediction horizon =",
  time_horizon,
  "months\n"
)

## ===============================
## 3. HELPER FUNCTIONS
## ===============================
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
  
  sd(
    x,
    na.rm = TRUE
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

ordinary_boot_idx_surv <- function(data) {
  sample(
    seq_len(nrow(data)),
    size = nrow(data),
    replace = TRUE
  )
}