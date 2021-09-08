
library(conflicted)
library(doParallel)
library(readr)
library(tidymodels)
library(vip)
library(xgboost)

tidymodels_prefer()
conflict_prefer("vi", "vip")
options(tidymodels.dark = TRUE)

data("iris")

### Split the dataset into training and test sets

set.seed(123)

iris_split <- initial_split(iris, prop = 3/4, strata = Species)
iris_train <- training(iris_split)
iris_test <- testing(iris_split)

### Create a recipe for the preprocessing steps on the training data

iris_rec <- recipe(Species ~ ., data = iris_train)

### Create a model specification for an XgBoost model 
### where we tune all the hyper-parameters keeping  number of trees fixed at 1000

iris_spec <-
  boost_tree(
    trees = 500,
    tree_depth = tune(),
    min_n = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    mtry = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

### Let's put evrything in a workflow

iris_flow <-
  workflow() %>%
  add_recipe(iris_rec) %>%
  add_model(iris_spec)

### Create a random grid of size 25 hyperparameters
set.seed(456)

iris_grid <-
  grid_latin_hypercube(
    tree_depth(),
    min_n(),
    loss_reduction(),
    sample_size = sample_prop(),
    finalize(mtry(), iris_train),
    learn_rate(),
    size = 20
  )

### Create a set of  bootstrap re-samples to use for tuning

set.seed(789)
iris_samples <- vfold_cv(iris_train, v = 5, strata = Species)

### Set up parallel processing

cl <- detectCores(logical = FALSE) - 1
registerDoParallel(cl)

### Perform grid search for optimal hyper-parameters on the grid defined earlier

iris_grid_res <-
  iris_flow %>%
  tune_grid(
    resamples = iris_samples,
    grid = iris_grid,
    metrics = metric_set(accuracy, roc_auc, mn_log_loss),
    control = control_grid(verbose = TRUE, save_pred = TRUE)
  )

### Bayesian Optimization to see if we can do any better (may be not)

iris_params <- 
  parameters(iris_flow) %>%
  update(mtry = mtry(c(1, 4)))

iris_bayes_res <-
  iris_flow %>%
  tune_bayes(
    resamples = iris_samples,
    param_info = iris_params,
    iter = 20,
    initial = iris_grid_res,
    control = control_bayes(no_improve = 10,
                            verbose = TRUE,
                            save_pred = TRUE)
  )

### Create the final model with the best hyper-parameter combination  using last_fit() to fit one final time to the training data and
### evaluate on the test set.

best_tune_params <- select_best(iris_bayes_res)

final_iris_mdl <-
  iris_flow %>%
  finalize_workflow(best_tune_params) %>%
  last_fit(iris_split)

### Plot the ROC curve

final_iris_mdl %>%
  collect_predictions() %>%
  roc_curve(Species, `.pred_setosa`:`.pred_virginica`) %>%
  ggplot(aes( 1 - specificity, sensitivity, color = .level)) +
  geom_abline(lty = 2, color = "black", size = 2) +
  geom_path(alpha = 0.5, size = 1.5) +
  coord_equal() +
  labs(
    x = "False Positive Rate",
    y = "True Positive Rate",
    title = "iris Classification Model ROC Curve",
    subtitle = "Predictions from iris Classification Model"
  ) +
  theme(
    text = element_text(
      face = "bold",
      size = 14),
    legend.position = "bottom"
  )








