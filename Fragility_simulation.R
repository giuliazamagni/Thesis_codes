################################################################################
## DECISION FRAGILITY SIMULATION STUDY
## Sample size, event rate, model complexity, and decision threshold
################################################################################

## ===============================
## 0. SETUP
## ===============================
library(data.table)
library(ranger)
library(xgboost)
library(ggplot2)

set.seed(20260720)

## ===============================
## 1. SIMULATION SETTINGS
## ===============================

N_grid <- c(
  1000,
  2000,
  5000,
  10000,
  50000
)

prevalence_grid <- c(
  0.10,
  0.20
)

dgp_grid <- c(
  "linear",
  "nonlinear"
)

threshold_grid <- c(
  0.05,
  0.08,
  0.10,
  0.15
)

models_to_run <- c(
  "logistic",
  "random_forest",
  "xgboost"
)

# Monte Carlo replicates
M <- 10

# Bootstrap replicates within each simulated dataset
B <- 100

# Bootstrap interval level
ci_level <- 0.95

# Random Forest parameters
rf_num_trees <- 300
rf_min_node_size <- 20
rf_max_depth <- 10

# XGBoost parameters
xgb_nrounds <- 150
xgb_max_depth <- 3
xgb_eta <- 0.05
xgb_subsample <- 0.80
xgb_colsample_bytree <- 0.80

# Output directory
output_directory <- "fragility_simulation_results"

if (!dir.exists(output_directory)) {
  dir.create(
    output_directory,
    recursive = TRUE
  )
}

## ===============================
## 2. CALIBRATE INTERCEPT
## ===============================

# Find the intercept required to obtain the target event rate
find_intercept <- function(
    linear_predictor_without_intercept,
    target_prevalence
) {
  
  objective_function <- function(intercept) {
    mean(
      plogis(
        intercept +
          linear_predictor_without_intercept
      )
    ) -
      target_prevalence
  }
  
  uniroot(
    objective_function,
    interval = c(-20, 20),
    tol = 1e-10
  )$root
}

## ===============================
## 3. GENERATE PREDICTORS
## ===============================

generate_predictors <- function(N) {
  
  # Correlated continuous predictors
  x1 <- rnorm(N)
  
  x2 <- 0.40 * x1 +
    sqrt(1 - 0.40^2) * rnorm(N)
  
  x3 <- rnorm(N)
  
  # Binary predictors
  x4 <- rbinom(
    N,
    size = 1,
    prob = 0.35
  )
  
  x5 <- rbinom(
    N,
    size = 1,
    prob = 0.20
  )
  
  # Additional continuous predictors
  x6 <- runif(
    N,
    min = -1,
    max = 1
  )
  
  x7 <- rnorm(N)
  
  # Noise predictors with no true effect
  x8 <- rnorm(N)
  
  x9 <- rbinom(
    N,
    size = 1,
    prob = 0.50
  )
  
  x10 <- runif(N)
  
  data.frame(
    x1 = x1,
    x2 = x2,
    x3 = x3,
    x4 = x4,
    x5 = x5,
    x6 = x6,
    x7 = x7,
    x8 = x8,
    x9 = x9,
    x10 = x10
  )
}

## ===============================
## 4. DATA-GENERATING MECHANISMS
## ===============================

calculate_true_linear_predictor <- function(
    X,
    dgp
) {
  
  if (dgp == "linear") {
    
    # Linear additive DGP
    eta <- (
      0.70 * X$x1 -
        0.50 * X$x2 +
        0.40 * X$x3 +
        0.60 * X$x4 +
        0.50 * X$x5 -
        0.40 * X$x6 +
        0.30 * X$x7
    )
    
  } else if (dgp == "nonlinear") {
    
    # Non-linear DGP including interactions,
    # quadratic, threshold, and sinusoidal effects
    eta <- (
      0.55 * X$x1 -
        0.35 * X$x2 +
        0.30 * X$x3 +
        0.45 * X$x4 +
        0.35 * X$x5 +
        0.80 * X$x1 * X$x4 +
        0.65 * X$x2 * X$x3 +
        0.70 * (X$x6^2 - mean(X$x6^2)) +
        0.75 * as.numeric(X$x7 > 0.50) +
        0.45 * sin(pi * X$x10)
    )
    
  } else {
    
    stop(
      "Unknown DGP: ",
      dgp
    )
  }
  
  eta
}

## ===============================
## 5. SIMULATE DATASET
## ===============================

simulate_dataset <- function(
    N,
    target_prevalence,
    dgp
) {
  
  X <- generate_predictors(N)
  
  eta_without_intercept <-
    calculate_true_linear_predictor(
      X,
      dgp
    )
  
  intercept <- find_intercept(
    linear_predictor_without_intercept =
      eta_without_intercept,
    target_prevalence =
      target_prevalence
  )
  
  true_probability <- plogis(
    intercept +
      eta_without_intercept
  )
  
  outcome <- rbinom(
    N,
    size = 1,
    prob = true_probability
  )
  
  data.frame(
    id = seq_len(N),
    outcome = outcome,
    true_probability = true_probability,
    X
  )
}

## ===============================
## 6. PREDICTOR MATRIX
## ===============================

predictor_names <- paste0(
  "x",
  1:10
)

make_predictor_matrix <- function(data) {
  
  matrix_data <- as.matrix(
    data[
      ,
      predictor_names,
      drop = FALSE
    ]
  )
  
  storage.mode(matrix_data) <- "double"
  
  matrix_data
}

## ===============================
## 7. LOGISTIC REGRESSION
## ===============================

fit_predict_logistic <- function(
    bootstrap_data,
    original_data
) {
  
  formula_logistic <- as.formula(
    paste(
      "outcome ~",
      paste(
        predictor_names,
        collapse = " + "
      )
    )
  )
  
  fitted_model <- suppressWarnings(
    glm(
      formula = formula_logistic,
      data = bootstrap_data,
      family = binomial(),
      control = glm.control(
        maxit = 50
      )
    )
  )
  
  predictions <- suppressWarnings(
    predict(
      fitted_model,
      newdata = original_data,
      type = "response"
    )
  )
  
  predictions <- pmin(
    pmax(
      predictions,
      1e-6
    ),
    1 - 1e-6
  )
  
  as.numeric(predictions)
}

## ===============================
## 8. RANDOM FOREST
## ===============================

fit_predict_random_forest <- function(
    bootstrap_data,
    original_data
) {
  
  rf_formula <- as.formula(
    paste(
      "factor(outcome) ~",
      paste(
        predictor_names,
        collapse = " + "
      )
    )
  )
  
  fitted_model <- ranger(
    formula = rf_formula,
    data = bootstrap_data,
    probability = TRUE,
    num.trees = rf_num_trees,
    min.node.size = rf_min_node_size,
    max.depth = rf_max_depth,
    mtry = max(
      1L,
      floor(
        sqrt(
          length(predictor_names)
        )
      )
    ),
    replace = TRUE,
    sample.fraction = 0.80,
    respect.unordered.factors = "order",
    num.threads = 1,
    seed = sample.int(
      .Machine$integer.max,
      1
    )
  )
  
  predictions <- predict(
    fitted_model,
    data = original_data,
    num.threads = 1
  )$predictions
  
  # ranger returns one probability column per class
  if (is.matrix(predictions)) {
    
    if ("1" %in% colnames(predictions)) {
      predictions <- predictions[, "1"]
    } else {
      predictions <- predictions[
        ,
        ncol(predictions)
      ]
    }
  }
  
  predictions <- pmin(
    pmax(
      predictions,
      1e-6
    ),
    1 - 1e-6
  )
  
  as.numeric(predictions)
}

## ===============================
## 9. XGBOOST
## ===============================

fit_predict_xgboost <- function(
    bootstrap_data,
    original_data
) {
  
  X_bootstrap <- make_predictor_matrix(
    bootstrap_data
  )
  
  X_original <- make_predictor_matrix(
    original_data
  )
  
  y_bootstrap <- bootstrap_data$outcome
  
  dtrain <- xgb.DMatrix(
    data = X_bootstrap,
    label = y_bootstrap
  )
  
  fitted_model <- xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "logloss",
      max_depth = xgb_max_depth,
      eta = xgb_eta,
      subsample = xgb_subsample,
      colsample_bytree =
        xgb_colsample_bytree,
      min_child_weight = 10,
      lambda = 1,
      alpha = 0,
      nthread = 1
    ),
    data = dtrain,
    nrounds = xgb_nrounds,
    verbose = 0
  )
  
  predictions <- predict(
    fitted_model,
    newdata = X_original
  )
  
  predictions <- pmin(
    pmax(
      predictions,
      1e-6
    ),
    1 - 1e-6
  )
  
  as.numeric(predictions)
}

## ===============================
## 10. MODEL DISPATCHER
## ===============================

fit_and_predict_model <- function(
    model_name,
    bootstrap_data,
    original_data
) {
  
  if (model_name == "logistic") {
    
    fit_predict_logistic(
      bootstrap_data,
      original_data
    )
    
  } else if (
    model_name == "random_forest"
  ) {
    
    fit_predict_random_forest(
      bootstrap_data,
      original_data
    )
    
  } else if (
    model_name == "xgboost"
  ) {
    
    fit_predict_xgboost(
      bootstrap_data,
      original_data
    )
    
  } else {
    
    stop(
      "Unknown model: ",
      model_name
    )
  }
}

## ===============================
## 11. DECISION FRAGILITY
## ===============================

calculate_fragility <- function(
    prediction_matrix,
    thresholds,
    ci_level = 0.95
) {
  
  alpha <- 1 - ci_level
  
  lower_quantile <- alpha / 2
  upper_quantile <- 1 - alpha / 2
  
  # Rows = individuals
  # Columns = bootstrap replicates
  lower_bound <- apply(
    prediction_matrix,
    MARGIN = 1,
    FUN = quantile,
    probs = lower_quantile,
    na.rm = TRUE,
    names = FALSE,
    type = 7
  )
  
  upper_bound <- apply(
    prediction_matrix,
    MARGIN = 1,
    FUN = quantile,
    probs = upper_quantile,
    na.rm = TRUE,
    names = FALSE,
    type = 7
  )
  
  result <- lapply(
    thresholds,
    function(threshold) {
      
      fragile <- (
        lower_bound <= threshold &
          upper_bound >= threshold
      )
      
      data.frame(
        threshold = threshold,
        
        fragile_n = sum(
          fragile,
          na.rm = TRUE
        ),
        
        fragility = mean(
          fragile,
          na.rm = TRUE
        ),
        
        mean_interval_width = mean(
          upper_bound -
            lower_bound,
          na.rm = TRUE
        )
      )
    }
  )
  
  rbindlist(result)
}

## ===============================
## 12. PROGRESS SETTINGS
## ===============================

total_progress_steps <- (
  length(N_grid) *
    length(prevalence_grid) *
    length(dgp_grid) *
    M *
    B *
    length(models_to_run)
)

cat(
  "\nStarting simulation\n",
  "Total model fits: ",
  total_progress_steps,
  "\n\n",
  sep = ""
)

progress_bar <- txtProgressBar(
  min = 0,
  max = total_progress_steps,
  initial = 0,
  style = 3,
  width = 60
)

progress_step <- 0L

## ===============================
## 13. RESULT CONTAINERS
## ===============================

all_fragility_results <- list()
simulation_diagnostics <- list()

result_counter <- 0L
diagnostic_counter <- 0L

## ===============================
## 14. MAIN SIMULATION LOOP
## ===============================

simulation_start_time <- Sys.time()

for (dgp in dgp_grid) {
  
  for (
    target_prevalence in prevalence_grid
  ) {
    
    for (N in N_grid) {
      
      for (
        replicate_id in seq_len(M)
      ) {
        
        # Reproducible seed for each scenario
        scenario_seed <- (
          1000000 +
            match(
              dgp,
              dgp_grid
            ) * 100000 +
            round(
              target_prevalence * 1000
            ) * 100 +
            match(
              N,
              N_grid
            ) * 10 +
            replicate_id
        )
        
        set.seed(scenario_seed)
        
        original_data <- simulate_dataset(
          N = N,
          target_prevalence =
            target_prevalence,
          dgp = dgp
        )
        
        observed_prevalence <- mean(
          original_data$outcome
        )
        
        observed_events <- sum(
          original_data$outcome
        )
        
        # One N x B prediction matrix
        # for each model.
        # With N = 50,000 and B = 100,
        # each matrix requires about 40 MB.
        prediction_matrices <- lapply(
          models_to_run,
          function(model_name) {
            matrix(
              NA_real_,
              nrow = N,
              ncol = B
            )
          }
        )
        
        names(prediction_matrices) <-
          models_to_run
        
        for (
          bootstrap_id in seq_len(B)
        ) {
          
          bootstrap_indices <- sample.int(
            n = N,
            size = N,
            replace = TRUE
          )
          
          bootstrap_data <- original_data[
            bootstrap_indices,
            ,
            drop = FALSE
          ]
          
          for (
            model_name in models_to_run
          ) {
            
            model_predictions <- tryCatch(
              
              fit_and_predict_model(
                model_name =
                  model_name,
                bootstrap_data =
                  bootstrap_data,
                original_data =
                  original_data
              ),
              
              error = function(e) {
                
                warning(
                  paste0(
                    "Error: DGP=", dgp,
                    ", prevalence=",
                    target_prevalence,
                    ", N=", N,
                    ", replicate=",
                    replicate_id,
                    ", bootstrap=",
                    bootstrap_id,
                    ", model=",
                    model_name,
                    ": ",
                    conditionMessage(e)
                  )
                )
                
                rep(
                  NA_real_,
                  N
                )
              }
            )
            
            prediction_matrices[
              [model_name]
            ][
              ,
              bootstrap_id
            ] <- model_predictions
            
            progress_step <-
              progress_step + 1L
            
            setTxtProgressBar(
              progress_bar,
              progress_step
            )
            
            flush.console()
          }
        }
        
        ## Decision fragility by model
        for (
          model_name in models_to_run
        ) {
          
          model_fragility <-
            calculate_fragility(
              prediction_matrix =
                prediction_matrices[
                  [model_name]
                ],
              thresholds =
                threshold_grid,
              ci_level =
                ci_level
            )
          
          model_fragility[
            ,
            `:=`(
              dgp = dgp,
              target_prevalence =
                target_prevalence,
              observed_prevalence =
                observed_prevalence,
              events =
                observed_events,
              N = N,
              replicate =
                replicate_id,
              model =
                model_name
            )
          ]
          
          result_counter <-
            result_counter + 1L
          
          all_fragility_results[
            [result_counter]
          ] <- model_fragility
        }
        
        diagnostic_counter <-
          diagnostic_counter + 1L
        
        simulation_diagnostics[
          [diagnostic_counter]
        ] <- data.table(
          dgp = dgp,
          target_prevalence =
            target_prevalence,
          observed_prevalence =
            observed_prevalence,
          events =
            observed_events,
          N = N,
          replicate =
            replicate_id
        )
        
        rm(prediction_matrices)
        
        gc(
          verbose = FALSE
        )
      }
    }
  }
}

close(progress_bar)

simulation_end_time <- Sys.time()

cat(
  "\nSimulation completed\n",
  "Duration: ",
  round(
    as.numeric(
      difftime(
        simulation_end_time,
        simulation_start_time,
        units = "mins"
      )
    ),
    2
  ),
  " minutes\n\n",
  sep = ""
)

## ===============================
## 15. RAW RESULTS
## ===============================

fragility_results <- rbindlist(
  all_fragility_results,
  use.names = TRUE,
  fill = TRUE
)

setcolorder(
  fragility_results,
  c(
    "dgp",
    "target_prevalence",
    "N",
    "replicate",
    "model",
    "threshold",
    "fragile_n",
    "fragility",
    "mean_interval_width",
    "observed_prevalence",
    "events"
  )
)

diagnostic_results <- rbindlist(
  simulation_diagnostics
)

fwrite(
  fragility_results,
  file.path(
    output_directory,
    "fragility_raw_results.csv"
  )
)

fwrite(
  diagnostic_results,
  file.path(
    output_directory,
    "simulation_diagnostics.csv"
  )
)

## ===============================
## 16. SUMMARY BY MODEL
## ===============================

fragility_summary <- fragility_results[
  ,
  .(
    mean_fragility = mean(
      fragility,
      na.rm = TRUE
    ),
    
    sd_fragility = sd(
      fragility,
      na.rm = TRUE
    ),
    
    median_fragility = median(
      fragility,
      na.rm = TRUE
    ),
    
    q025_fragility = quantile(
      fragility,
      probs = 0.025,
      na.rm = TRUE
    ),
    
    q975_fragility = quantile(
      fragility,
      probs = 0.975,
      na.rm = TRUE
    ),
    
    mean_interval_width = mean(
      mean_interval_width,
      na.rm = TRUE
    ),
    
    mean_observed_prevalence = mean(
      observed_prevalence,
      na.rm = TRUE
    ),
    
    mean_events = mean(
      events,
      na.rm = TRUE
    )
  ),
  
  by = .(
    dgp,
    target_prevalence,
    N,
    model,
    threshold
  )
]

fwrite(
  fragility_summary,
  file.path(
    output_directory,
    "fragility_summary_by_model.csv"
  )
)

## ===============================
## 17. FRAGILITY VS SAMPLE SIZE
## ===============================

fragility_summary[
  ,
  threshold_factor :=
    factor(threshold)
]

fragility_plot <- ggplot(
  fragility_summary,
  aes(
    x = N,
    y = mean_fragility,
    color = model,
    group = model
  )
) +
  geom_line(
    linewidth = 1
  ) +
  scale_color_discrete(
    labels = c(
      logistic = "Logistic",
      random_forest = "Random Forest",
      xgboost = "XGBoost"
    )
  ) +
  scale_x_continuous(
    breaks = N_grid
  ) +
  facet_grid(
    threshold_factor ~
      dgp +
      target_prevalence,
    labeller = labeller(
      threshold_factor =
        label_value,
      dgp = c(
        linear = "Linear DGP",
        nonlinear =
          "Non-linear DGP"
      ),
      target_prevalence = c(
        `0.1` =
          "Event rate = 10%",
        `0.2` =
          "Event rate = 20%"
      )
    )
  ) +
  labs(
    x = "Sample size",
    y = "Mean decision fragility",
    color = ""
  ) +
  theme_bw() +
  theme(
    axis.text =
      element_text(size = 12),
    axis.title =
      element_text(size = 13),
    axis.text.x =
      element_text(
        angle = 60,
        hjust = 1
      ),
    legend.position = "bottom",
    legend.text =
      element_text(size = 13),
    strip.text =
      element_text(size = 13)
  )

print(fragility_plot)

ggsave(
  filename = file.path(
    output_directory,
    "fragility_vs_N.png"
  ),
  plot = fragility_plot,
  width = 10,
  height = 10,
  dpi = 600
)

## ===============================
## 18. PRINT SUMMARY
## ===============================
print(fragility_summary)