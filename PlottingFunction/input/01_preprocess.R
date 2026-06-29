
# load data and merge -------------

library(readr)
library(readxl)

#sample info
batch_design_signaling_lipid_1 <- read_excel("data/0_sample_info/batch design_signaling lipid 1.xlsx")
View(batch_design_signaling_lipid_1)
sample_info <- batch_design_signaling_lipid_1[, c(1:5, 7)]
colnames(sample_info)[6] <- "injectionID"
colnames(sample_info)[1] <- "cell"

#load low ph
low_ph <- read_excel("data/1_mzquality/lowpH_after_mzquality.xlsx")
View(low_ph)

low_ph$...1 <- substr(low_ph$...1, 9, nchar(low_ph$...1) - 3)
colnames(low_ph) <- sub("^.{4}", "", colnames(low_ph))
colnames(low_ph)[1] <- "injectionID"
low_ph$injectionID <- as.numeric(low_ph$injectionID)
low_ph <- low_ph[,-2]

#load high ph
high_ph <- read_excel("data/1_mzquality/highpH_after_mzquality.xlsx")
high_ph$...1 <- substr(high_ph$...1, 9, nchar(high_ph$...1) - 3)
colnames(high_ph) <- sub("^.{4}", "", colnames(high_ph))
colnames(high_ph)[1] <- "injectionID"
high_ph$injectionID <- as.numeric(high_ph$injectionID)
high_ph <- high_ph[,-2]


#combine data 
# Step 1: Find overlapping columns
overlapping_columns <- intersect(colnames(low_ph[,2:29]), colnames(high_ph[,2:16]))

# Step 2: Remove overlapping columns from the second dataset
high_ph <- high_ph[, !colnames(high_ph) %in% overlapping_columns]

# Step 3: Merge the datasets by injectionID
library(dplyr)
all_target <- left_join(low_ph, high_ph, by = "injectionID")

colnames(all_target)

df <- all_target

df_n <- data.frame(lapply(all_target, as.numeric))


# colnames(df_n)[1] <- ""
# df_n[is.na(df_n)] <- ""
# 
# write.csv(df_n, file = "df_n.csv", row.names = FALSE)
colnames(df_n)[1] <- "count"
merged_df <- merge(sample_info, df_n, by = "count", all.x = TRUE)
write.csv(merged_df, file = "data/2_file_for_use/merged.csv", row.names = FALSE)

merged_df <- read.csv("merged.csv", header = TRUE)

# # exclude the rows that injectionID = 1/8/12/15/30/31/41/49
# df_n_filter <- df_n[-c(1,8,12,15,30,31,41,49), ]

# imputation
# Load necessary library
library(BiocManager)

# BiocManager::install("pcaMethods")
# BiocManager::install("impute")
# 
# install.packages("imputeLCMD")
# 
# update.packages(ask = FALSE)

library(impute)
library(imputeLCMD)
# impute.QRILC

# # turn all the columns in test1_10 into numeric
# test1_10 <- data.frame(lapply(test1_10, as.numeric))

# Step 1: Calculate the missing rate for each target
missing_rate <- colMeans(is.na(df_n[,-1]))

# Step 2: Remove targets with more than 20% missing data
df_filtered <- df_n[, c(TRUE, missing_rate <= 0.20)]

# Step 3: Impute the remaining missing data using QRILC
# Extract the data to be imputed (excluding the SampleID column)
data_to_impute <- df_filtered[, -1]
# Ensure the data is numeric
data_to_impute <- as.data.frame(lapply(data_to_impute, as.numeric))

# Example: log2-transform if not already done
data_log <- log2(data_to_impute)  

# QRILC only accepts matrices

data_matrix <- as.matrix(data_log)
storage.mode(data_matrix) <- "numeric"
data_matrix[!is.finite(data_matrix)] <- NA
data_matrix <- t(data_matrix)


set.seed(123)
# Impute using MinProb
imputed <- imputeLCMD::impute.QRILC(data_matrix, tune.sigma = 1)[[1]]

# Check result
str(imputed)  # should be a matrix
#transpose back to original format
imputed <- t(imputed)

# imputated_df_linear <- 2^imputed
imputed_df <- cbind(count = df_filtered[,1], imputed)

# imputated_df <- read.csv("imputated.csv", header = TRUE)
colnames(imputed_df)[2:12] <- sub("^.{1}", "", colnames(imputed_df)[2:12])
imputed_df <- as.data.frame(imputed_df)

merged_df_imputed <- merge(sample_info, imputed_df, by = "count")

# check any na in dataframe
anyNA(merged_df_imputed)

write.csv(merged_df_imputed, file = "data/2_file_for_use/merged_df_after_imputation.csv", row.names = FALSE)

# check samples to remove
library(readr)
# load the data 
df_full <- read.csv("data/2_file_for_use/merged_df_after_imputation.csv")
df_group <- read.csv("data/2_file_for_use/df_group.csv")
# remove outliner
df_PCA_pre <- cbind(Group = df_group$group, df_full[-13,])

# check the variance of each column
apply(df_PCA_pre[8:49], 2, function(x) sd(x, na.rm = TRUE))
zero_var <- sapply(df_PCA_pre[8:49], function(x) sd(x, na.rm = TRUE) == 0)
# remove the targets that have zero variance
df_PCA_pre <- df_PCA_pre[, -c(25,31:33)]
colnames(df_PCA_pre)[6] <- "repeats"

write.csv(df_PCA_pre, file = "data/2_file_for_use/df_final.csv", row.names = FALSE)

df <- read.csv("data/2_file_for_use/df_final.csv", header = TRUE)

# long format
df_wide <- df[,-c(1,2,6,7)]

library(dplyr)
library(tidyr)

df_long <- df_wide %>%
  pivot_longer(
    cols = -c(cell, group, time),
    names_to = "compound",
    values_to = "value"
  )

# line plot
library(dplyr)
library(ggplot2)

plot_df <- df_long %>%
  mutate(
    line_group = case_when(
      cell == "HCC1143" & group == "Control"   ~ "1143control",
      cell == "HCC1143" & group == "Migration" ~ "1143migration",
      cell == "HCC38"   & group == "Control"   ~ "38control",
      cell == "HCC38"   & group == "Migration" ~ "38migration",
      cell == "media" | group == "media"       ~ "media",
      TRUE ~ NA_character_
    ),
    time = factor(time, levels = c(0, 12, 18, 22))
  ) %>%
  group_by(compound, time, line_group) %>%
  summarise(
    mean_value = mean(value, na.rm = TRUE),
    se_value   = sd(value, na.rm = TRUE) / sqrt(sum(!is.na(value))),
    .groups = "drop"
  )

library(ggplot2)
library(dplyr)

plot_df <- plot_df %>%
  mutate(time = as.numeric(as.character(time)))

all_compounds <- unique(plot_df$compound)

for (cmp in all_compounds) {
  
  df_sub <- plot_df %>%
    filter(compound == cmp)
  
  p <- ggplot(df_sub,
              aes(x = time, y = mean_value, color = line_group, group = line_group)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2) +
    scale_x_continuous(
      breaks = c(0, 12, 18, 22),
      limits = c(0, 22),
      expand = c(0, 0)
    ) +
    theme_classic() +
    labs(
      title = cmp,
      x = "Time",
      y = "Mean value",
      color = "Condition"
    )
  
  file_name <- paste0("plot/all_", gsub("[^A-Za-z0-9_\\-]", "_", cmp), ".png")
  
  ggsave(file_name, plot = p, width = 6, height = 4.5, dpi = 300)
}



# PCA --------------
df <- read.csv("data/2_file_for_use/df_final.csv", header = TRUE)

df_PCA <- df[, c(1,8:45)]

#turn all columns to numeric
df_PCA<- data.frame(lapply(df_PCA[2:39], as.numeric))

lipid_pca <- prcomp(df_PCA, scale. = TRUE)

summary(lipid_pca)

####draw plot to show proportion of variance
# Extract eigenvalues (squared singular values) from PCA results
eigenvalues <- (lipid_pca$sdev)^2

# Calculate proportion of variance explained by each principal component
variance_explained <- eigenvalues / sum(eigenvalues)

# Calculate cumulative proportion of variance explained
cumulative_variance <- cumsum(variance_explained)

# Create a dataframe for plotting
variance_data <- data.frame(Principal_Component = 1:length(variance_explained),
                            Variance_Explained = variance_explained,
                            Cumulative_Variance = cumulative_variance)

# Plot proportion of variance explained by principal components
library(ggplot2)

ggplot(variance_data, aes(x = Principal_Component, y = Variance_Explained)) +
  geom_line() +
  geom_point() +
  labs(title = "Proportion of Variance Explained by Principal Components_",
       x = "Principal Component",
       y = "Proportion of Variance Explained") +
  theme_minimal()

###plots for pca results
plot(lipid_pca,type="lines")

###############################################################draw pca plot(2D)

# load the package
library(ggplot2)

scores <- data.frame(PC1 = lipid_pca$x[,1], PC2 = lipid_pca$x[,2], cell = df_PCA_pre$cell, group = df_PCA_pre$group,time = df_PCA_pre$time, sample = df_PCA_pre$count)

scores$group <- as.factor(scores$group)
scores$cell <- as.factor(scores$cell)
scores$time <- as.factor(scores$time)


# Identify the points with the highest PC1 and PC2 values
highest_PC1 <- scores[which.max(scores$PC2), ]
highest_PC2 <- scores[which.min(scores$PC1), ]

# Combine these points into a single data frame
outliers <- rbind(highest_PC1, highest_PC2)
library(ggplot2)
ggplot(scores, aes(x = PC1, y = PC2, color = cell, shape = time)) +
  geom_point(size = 3) + # Use hollow circles
  stat_ellipse(aes(group = cell), type = "t", level = 0.95) + # This adds the ellipses
  theme_classic() +
  labs(title = "PCA Plot by cell line", x = "PC1 (39.99%)", y = "PC2 (10.54%)") +
  theme(legend.position = "right") +
  scale_color_discrete(name = "cell line") 
# +
#   geom_point(data = outliers, aes(x = PC1, y = PC2), size = 4) + # Highlight outliers
#   geom_text(data = outliers, aes(label = sample), vjust = -1, hjust = 1, color = "red")


