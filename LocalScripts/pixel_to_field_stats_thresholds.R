# Load require packages
library(tidyverse)

# Import dataset with V2 Model 1 Predictions
v2m1_preds <- read_csv("Data/V2_Model1_predictions_score.csv", 
                       col_types = cols(X1 = col_skip()))

# Change NaNs to NAs
v2m1_preds[v2m1_preds == "NaN"] <- NA

# Combine cols: pred_test, pred_train. New col called "pred"
v2m1_preds_1 <- v2m1_preds %>%
        mutate(pred = coalesce(pred_test, pred_train), .keep = "unused")

# Aggregate Pixels to field level: 
# grouped by field, calculate sum of num_pixels, mean, 25th, 50th, 75th, 90th percentiles
# Keep cols: "label", "field_id", "border_flag"
# Drop all other cols
v2m1_preds_summ <- v2m1_preds_1 %>%
        group_by(field_id) %>%
        summarize(sum_n_pixels = sum(NumImages),
                  mean_pred = mean(pred),
                  p25_pred = quantile(pred, .25),
                  p50_pred = quantile(pred, .50),
                  p75_pred = quantile(pred, .75),
                  p90_pred = quantile(pred, .90))

# df of unique field_id with label
field_label <- v2m1_preds_1 %>%
        select(field_id, label) %>%
        unique()

# join mean/quantile summaries with field label df
v2m1_preds_summ_l <- v2m1_preds_summ %>%
        left_join(field_label, by = "field_id")


# Create function that calculates fit for every value of percentile from .1 to .99, takes measure as input
calculate_fits <- function(measure) {
        #initialize df
        df <- data.frame(percentile = numeric(), fit = numeric())
        percentile <- .01
        #loop over percentile from 0.01 to 0.99 in increments of 0.01
        while(percentile < 1) {
                        v2m1_preds_summ_l$measure_pred_label <- ifelse(measure > percentile, 1, 0)
                        v2m1_preds_summ_l$measure_true <- ifelse(v2m1_preds_summ_l$measure_pred_label == v2m1_preds_summ_l$label, 1, 0)
                        measure_fit <- sum(v2m1_preds_summ_l$measure_true)/length(v2m1_preds_summ_l$measure_true)
                        vec <- data.frame("percentile" = percentile, "measure_fit" = measure_fit)
                        df <- rbind(df, vec)
                        percentile <- percentile + 0.01
        }
        df
}
        
# calculate_fits for each measure and assign result to a df combining all the measures' fits
mean <- calculate_fits(v2m1_preds_summ$mean_pred)
p25 <- calculate_fits(v2m1_preds_summ$p25_pred)
p50 <- calculate_fits(v2m1_preds_summ$p50_pred)
p75 <- calculate_fits(v2m1_preds_summ$p75_pred)
p90 <- calculate_fits(v2m1_preds_summ$p90_pred)

# Join all dfs with threshold increments into one df
fits <- list(mean,p25,p50, p75, p90) %>%
        Reduce(function(dtf1,dtf2) left_join(dtf1,dtf2, by = "percentile"), .)

colnames(fits) <- c("percentile", "mean_fit", "p25_fit", "p50_fit", "p75_fit", "p90_fit")

# Make a graph of fit by percentile, colored by measure
fits_long <- pivot_longer(fits, c(-percentile), names_to = "vars")

fits_long %>%
        ggplot(aes(x = percentile, y = value, color = vars)) + geom_line() + labs(title = "Percentile Threshold vs Prediction Accuracy on ", y = "Prediction Accuracy")

# keep max fit values for each measure and threshold
which_max_indexed <- function(measure) {
        which(measure == max(measure))
}
indices <- list(fits$mean_fit, fits$p25_fit, fits$p50_fit, fits$p75_fit, fits$p90_fit)
threshold_medians <- map(map(indices, which_max_indexed), median)
set_names(threshold_medians, c("mean_fit", "p25_fit", "p50_fit", "p75_fit", "p90_fit"))



