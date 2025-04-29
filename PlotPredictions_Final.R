


### Author: Andrew Valesano
### Project: STIC machine learning
### Purpose: Make scatterplots and ROC curves for patient-level predictions.
### Note: For category_stic_stil, STIC_STIL is 0 and benign is 1.

# =========================== Modules, working directory, and metadata ============================

library(tidyverse)
library(plotROC)
library(pROC)
library(precrec)
library(patchwork)
library(wesanderson)
library(irr)
library(rsample)
library(ROCR)
library(purrr)
library(yardstick)
library(cowplot)

palette <- wes_palette("Zissou1")
palette_2 <- c("#78B7C5", "#1A346C", "#F54B19")

setwd("Projects/STIC_ML")

metadata <- read_csv("data/metadata_rescanned.csv")

# ============================== Load the initial holdout dataset results =======================================

patient_predictions <- read_csv("data/model_output/results_segment_rois_amil/training/00003-attention_mil-category_stic_stil/predictions.csv")

patient_predictions <- left_join(patient_predictions, metadata, by = "slide")

# ===================================== Summary statistics and statistical testing ===============================

# pos vs neg
patient_predictions %>%
  group_by(category_stic_stil) %>%
  summarize(median = median(y_pred0))

patient_predictions_pos <- filter(patient_predictions, category_stic_stil == "STIC_STIL")
patient_predictions_neg <- filter(patient_predictions, category_stic_stil == "benign")

t.test(patient_predictions_pos$y_pred0, patient_predictions_neg$y_pred0)

# STIC vs STIL
patient_predictions %>%
  group_by(diagnosis) %>%
  summarize(median = median(y_pred0))

patient_predictions_stic <- filter(patient_predictions, diagnosis == "STIC")
patient_predictions_stil <- filter(patient_predictions, diagnosis == "STIL")

t.test(patient_predictions_stic$y_pred0, patient_predictions_stil$y_pred0)

# Threshold-dependent metrics

threshold = 0.3

eval <- patient_predictions

tp <- sum(eval$y_pred0 >= threshold & eval$category_stic_stil == "STIC_STIL")
fp <- sum(eval$y_pred0 >= threshold & eval$category_stic_stil == "benign")
tn <- sum(eval$y_pred0 < threshold & eval$category_stic_stil == "benign")
fn <- sum(eval$y_pred0 < threshold & eval$category_stic_stil == "STIC_STIL")

sensitivity <- tp / (tp + fn)
specificity <- tn / (tn + fp)
precision <- tp / (tp + fp)
accuracy <- (tp + tn) / (tp + tn + fp + fn)
recall <- sensitivity
F1_score <- 2*((precision*recall) / precision + recall)

# =================================================== ROC on holdout =============================================================

roc.plot.holdout <- ggplot(patient_predictions, aes(m = y_pred0, d = 1 - y_true)) +
  geom_roc(labels = FALSE, n.cuts = 0) +
  theme_bw() + 
  xlab("False Positive Rate") + 
  ylab("True Positive Rate") +
  theme(text = element_text(size = 20)) +
  ggtitle("Performance on Holdout Dataset")
roc.plot.holdout

roc.plot.holdout.auc <- calc_auc(roc.plot.holdout)

roc.plot.holdout.auc.label <- paste0("AUC = ", as.character(round(roc.plot.holdout.auc$AUC, 2)))

# PDF, 10 by 12
roc.plot.holdout <- roc.plot.holdout + 
  annotate("text", x = 0.9, y = 0.1, label = roc.plot.holdout.auc.label, size = 10, color = palette[5])
roc.plot.holdout

# ============================================ Making ROC with confidence intervals ================================================


# Prep data
df <- patient_predictions %>%
  mutate(d = 1 - y_true, m = y_pred0) %>%
  select(m, d)

# Bootstrap sampling
set.seed(123)
boot_samples <- bootstraps(df, times = 1000)

# Function to compute ROC points
compute_roc <- function(data) {
  pred <- prediction(data$m, data$d)
  perf <- performance(pred, "tpr", "fpr")
  tibble(
    FP = perf@x.values[[1]],
    TP = perf@y.values[[1]]
  )
}

# Compute ROC curves for each sample
boot_roc <- boot_samples %>%
  mutate(roc = map(splits, ~ compute_roc(analysis(.x)))) %>%
  unnest(roc)

# Summarize confidence intervals across bootstraps
roc_ci <- boot_roc %>%
  group_by(FP) %>%
  summarise(
    TPR_mean = mean(TP, na.rm = TRUE),
    TPR_lower = quantile(TP, 0.025, na.rm = TRUE),
    TPR_upper = quantile(TP, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

# Bootstrapped ROC curves (in alpha-black)
boot_roc_plot <- ggplot() +
  geom_path(data = boot_roc, aes(x = FP, y = TP, group = id), color = "#1A346C", alpha = 0.05)

# Combine them
final_plot <- boot_roc_plot +
  geom_roc(data = patient_predictions, aes(m = y_pred0, d = 1 - y_true), 
           color = "#F54B19", size = 1.5, n.cuts = 0, labels = FALSE) +
  theme_bw() + 
  xlab("False Positive Rate") + 
  ylab("True Positive Rate") +
  theme(text = element_text(size = 20), panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank())
final_plot # PDF 8 by 8

# AUC
boot_samples <- bootstraps(patient_predictions, times = 1000)

boot_auc <- boot_samples %>%
  mutate(auc = map_dbl(splits, ~ {
    analysis(.x) %>%
      mutate(y_true = factor(y_true)) %>%
      roc_auc(y_true, y_pred0) %>%
      dplyr::pull(.estimate)
  }))

boot_auc_summary <- boot_auc %>%
  summarise(
    mean_auc = mean(auc),
    lower_ci = quantile(auc, 0.025),
    upper_ci = quantile(auc, 0.975)
  )

quantile(boot_auc$auc, c(0.025, 0.5, 0.975))  # 95% CI for AUC

# ================================================== ROC of k-folds ======================================================


patient_predictions_kfold1 <- read.csv("data/model_output/results_segment_rois_amil/training/00000-attention_mil-category_stic_stil/predictions.csv", stringsAsFactors = FALSE) %>% mutate(kfold = "k-fold 1")

patient_predictions_kfold2 <- read.csv("data/model_output/results_segment_rois_amil/training/00000-attention_mil-category_stic_stil/predictions.csv", stringsAsFactors = FALSE) %>% mutate(kfold = "k-fold 2")

patient_predictions_kfold3 <- read.csv("data/model_output/results_segment_rois_amil/training/00000-attention_mil-category_stic_stil/predictions.csv", stringsAsFactors = FALSE) %>% mutate(kfold = "k-fold 3")

patient_predictions_kfolds <- rbind(patient_predictions_kfold1, patient_predictions_kfold2, patient_predictions_kfold3) %>% 
  mutate(y_false = ifelse(y_true == 0, 1, 0))


roc.plot.kfolds <- ggplot(patient_predictions_kfolds, aes(m = y_pred0, d = y_false, color = kfold)) +
  geom_roc(labels = FALSE, n.cuts = 0) +
  theme_bw() + 
  xlab("False Positive Rate") + 
  ylab("True Positive Rate") +
  theme(legend.position = "none", text = element_text(size = 20)) +
  theme(axis.text.x = element_text(size = 10)) +
  scale_color_manual(values = c(palette[1], palette[3], palette[5])) +
  ggtitle("Cross-validation ROC") +
  facet_wrap(~kfold)
roc.plot.kfolds

# ============================================== Scatterplot of scores ========================================================


patient_predictions$diagnosis  <- factor(patient_predictions$diagnosis, levels = c("benign", "STIL", "STIC"))

prediction.scores.holdout <- ggplot(patient_predictions, aes(x = as.factor(diagnosis), y = y_pred0, color = diagnosis)) +
  geom_jitter(width = 0.2, size = 7, alpha = 0.8) +
  geom_boxplot(alpha = 0, color = "black") + 
  theme_bw() +
  theme(text = element_text(size = 20), legend.position = "none") +
  xlab("") +
  ylab("Prediction Score") +
  #ggtitle("Holdout Prediction Scores") +
  scale_color_manual(values = c(palette[1], palette[3], palette[5])) +
  geom_hline(yintercept = 0.3, col = "black", linetype = "dashed")
prediction.scores.holdout


# =================== Results for additional validation cohort =========================

new_predictions <- read.csv("data/model_output/results_segment_rois_amil/eval/00001-attention_mil/predictions.csv", stringsAsFactors = FALSE)
metadata_new <- read_csv("data/metadata_new_20250210.csv")

new_predictions_meta <- left_join(metadata_new, new_predictions, by = "slide")

new_predictions_meta$diagnosis  <- factor(new_predictions_meta$diagnosis, levels = c("benign", "STIL", "STIC"))

new_predictions_meta <- new_predictions_meta %>%
  group_by(diagnosis) %>%
  mutate(count = n())

# Prediction scores
prediction.scores.holdout.new <- ggplot(new_predictions_meta, aes(x = as.factor(diagnosis), y = y_pred0, color = diagnosis)) +
  geom_jitter(width = 0.2, size = 5, alpha = 0.8) +
  geom_boxplot(data = subset(new_predictions_meta, count > 2), alpha = 0, color = "black") + 
  theme_bw() +
  theme(text = element_text(size = 20), legend.position = "none") +
  xlab("") +
  ylab("Prediction Score") +
  #ggtitle("New Cohort Prediction Scores") +
  scale_color_manual(values = c(palette[1], palette[3], palette[5])) +
  geom_hline(yintercept = 0.3, col = "black", linetype = "dashed")
prediction.scores.holdout.new


# ROC curve and AUC
roc.plot.new <- ggplot(new_predictions_meta, aes(m = y_pred0, d = 1 - y_true, color = "#F54B19")) +
  geom_roc(labels = FALSE, n.cuts = 0, size = 1.5) +
  theme_bw() + 
  xlab("False Positive Rate") + 
  ylab("True Positive Rate") +
  theme(text = element_text(size = 20), legend.position = "none")
roc.plot.new

roc.plot.new <- ggplot(new_predictions_meta, aes(m = y_pred0, d = 1 - y_true)) +
  geom_roc(labels = FALSE, n.cuts = 0, size = 1.5) +
  theme_bw() + 
  xlab("False Positive Rate") + 
  ylab("True Positive Rate") +
  theme(text = element_text(size = 20), legend.position = "none")

# Get AUROC
roc.plot.new.auc <- calc_auc(roc.plot.new)

# Now bootstrap AUROC
boot_samples <- bootstraps(patient_predictions, times = 1000)

boot_auc <- boot_samples %>%
  mutate(auc = map_dbl(splits, ~ {
    df <- analysis(.x)
    
    if (length(unique(df$y_true)) < 2) {
      return(NA_real_)  # Skip if only one class is present
    }
    
    df %>%
      mutate(y_true = factor(y_true)) %>%
      roc_auc(y_true, y_pred0) %>%
      dplyr::pull(.estimate)
  }))

quantile(boot_auc$auc, c(0.025, 0.975))  # 95% CI for AUC


# =================== Results for reactive atypia cases =========================

metadata_atypical <- read_csv("data/metadata_atypical_eval.csv")

predictions_atypical <- read.csv("data/model_output/results_segment_rois_amil/eval/00003-attention_mil/predictions.csv", stringsAsFactors = FALSE)

predictions_atypical_meta <- left_join(metadata_atypical, predictions_atypical, by = "slide")

predictions_atypical_meta %>%
  summarize(median = median(y_pred0), stdev = sd(y_pred0))

predictions_atypical_meta_low <- filter(predictions_atypical_meta, y_pred0 <= 0.3)
nrow(predictions_atypical_meta_low) / nrow(predictions_atypical_meta)

prediction.scores.atypical <- ggplot(predictions_atypical_meta, aes(x = as.factor(diagnosis), y = y_pred0)) +
  geom_jitter(width = 0.2, size = 5, alpha = 0.8, color = "#1A346C") +
  geom_boxplot(data = predictions_atypical_meta, alpha = 0, color = "black", width = 0.4) + 
  theme_bw() +
  theme(text = element_text(size = 20), legend.position = "none") +
  xlab("") +
  ylab("Prediction Score") +
  geom_hline(yintercept = 0.3, col = "black", linetype = "dashed")
prediction.scores.atypical

# =========================== Plot UMAP ============================

umap_data <- read_csv("data/model_output/results_segment_rois_amil/training//mosaic/slidemap.csv")

umap_data_metadata <- left_join(umap_data, metadata, by = "slide")

umap_data_metadata$diagnosis  <- factor(umap_data_metadata$diagnosis, levels = c("benign", "STIL", "STIC"))

umap_data_metadata %>%
  ggplot(aes(x = x, y = 1 - y)) +
  geom_point() +
  facet_wrap(~diagnosis)

umap.plot.score <- ggplot(umap_data_metadata, aes(x = x, y = y, color = diagnosis)) +
  geom_point(alpha = 0.5) +
  theme_minimal() +
  facet_wrap(~diagnosis) +
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  #scale_colour_viridis_d(option = "inferno", name = "") +
  #scale_color_viridis_c(option = "viridis", name = "STIC-STIL Prediction Score") +
  #scale_color_manual(name = "Diagnosis", values = c(palette[1], palette[3], palette[5])) +
  scale_color_manual(name = "Diagnosis", values = c(palette_2[1], palette_2[2], palette_2[3])) +
  theme(text = element_text(size = 20)) +
  theme(legend.text = element_text(size = 20))
umap.plot.score

umap.plot.score.density <- ggplot(umap_data_metadata, aes(x = x, y = y, fill = diagnosis)) +
  stat_density_2d(geom = "polygon", aes(alpha = ..level..), contour = TRUE, n = 100, bins = 30) +
  scale_fill_manual(values = palette_2, name = "") +
  scale_alpha(range = c(0.2, 0.7), guide = 'none') +
  theme_bw() +
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  xlim(c(-0.05, 1.05)) +
  ylim(c(-0.05, 1.05)) +
  theme(panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank(), text = element_text(size = 20), legend.justification = c(0.95, 0.05), legend.position = c(0.95, 0.05), legend.box.background = element_blank())
umap.plot.score.density

# =================================== Making the figure ==================================

figure_2_cowplot <- plot_grid(final_plot, prediction.scores.holdout, labels = c('A', 'B'), label_size = 20, label_fontface = "bold") # 8 by 16

figure_2 <- final_plot | prediction.scores.holdout

ggsave(figure_2, filename = "results/Figure_2.pdf", device = cairo_pdf, width = 16, height = 8, units = "in")

ggsave(umap.plot.score.density, filename = "results/Figure_3A.pdf", device = cairo_pdf, width = 8, height = 8, units = "in")

figure_4_cowplot <- plot_grid(roc.plot.new, prediction.scores.holdout.new, prediction.scores.atypical, ncol = 3, rel_widths = c(2, 2, 1))

ggsave(figure_4_cowplot, filename = "results/Figure_4.pdf", device = cairo_pdf, width = 20, height = 8, units = "in")          
