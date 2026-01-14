# ============================================================================
# Time Series Analysis - Using Sample Data
# Alternative to API when endpoint is unavailable
# ============================================================================

cat("\n========== Time Series Learning - Sample Data ==========\n\n")

# Since the EIA API endpoint is currently unavailable,
# we'll use realistic sample energy data for learning time series forecasting

library(data.table)
library(ggplot2)
library(dplyr)

# Create sample electricity generation data
set.seed(42)

# Generate dates
start_date <- as.Date("2020-01-01")
end_date <- as.Date("2024-12-31")
dates <- seq(start_date, end_date, by = "month")

# Create realistic electricity generation data (in GWh)
# Includes trend, seasonality, and random variation
base_value <- 4000
trend <- seq(0, 500, length.out = length(dates))
seasonal <- 500 * sin(2 * pi * seq(0, 1, length.out = length(dates)))
random_noise <- rnorm(length(dates), mean = 0, sd = 100)
values <- base_value + trend + seasonal + random_noise

# Create data frame
df_raw <- data.frame(
  period = dates,
  respondent = "US48",
  value = values,
  type = "generation",
  stringsAsFactors = FALSE
)

# Convert to data.table
dt_clean <- as.data.table(df_raw)

cat("✓ Sample data created successfully!\n")
cat("Dimensions:", nrow(dt_clean), "rows x", ncol(dt_clean), "columns\n\n")

# ============================================================================
# STEP 5: Data Inspection
# ============================================================================
cat("========== DATA STRUCTURE ==========\n")
print(str(dt_clean))

cat("\n========== FIRST 10 ROWS ==========\n")
print(head(dt_clean, 10))

cat("\n========== SUMMARY STATISTICS ==========\n")
print(summary(dt_clean))

# ============================================================================
# STEP 6: Data Cleaning
# ============================================================================
cat("\n========== CLEANED DATA ==========\n")

# Identify columns
date_col <- "period"
value_col <- "value"

# Ensure proper data types
dt_clean[[date_col]] <- as.Date(dt_clean[[date_col]])
dt_clean[[value_col]] <- as.numeric(dt_clean[[value_col]])

# Sort by date
dt_clean <- dt_clean[order(get(date_col))]

print(head(dt_clean, 10))

# ============================================================================
# STEP 7: Time Series Validation
# ============================================================================
cat("\n========== TIME SERIES VALIDATION ==========\n")

dates <- dt_clean[[date_col]]
values <- dt_clean[[value_col]]

cat("Date Range: ", min(dates, na.rm = TRUE), " to ", max(dates, na.rm = TRUE), "\n")
cat("Total observations:", length(dates), "\n")

# Check for gaps
date_diffs <- diff(dates)
mode_diff <- as.numeric(names(sort(table(date_diffs), decreasing = TRUE)[1]))
cat("Most common interval (days):", mode_diff, "\n")
cat("Any gaps detected:", any(date_diffs > mode_diff), "\n")

# Check for duplicates
dup_dates <- sum(duplicated(dates))
cat("Duplicate dates:", dup_dates, "\n")

# Check for missing values
cat("Missing values in value column:", sum(is.na(values)), "\n")

# ============================================================================
# STEP 8: Create Exploratory Plots
# ============================================================================
cat("\n📈 Creating exploratory visualizations...\n\n")

plot_data <- data.frame(
  date = dt_clean[[date_col]],
  value = dt_clean[[value_col]]
)

# Remove NA values
plot_data <- plot_data[complete.cases(plot_data), ]

# Plot 1: Time Series
p1 <- ggplot(plot_data, aes(x = date, y = value)) +
  geom_line(color = "#2E86AB", linewidth = 0.7) +
  geom_point(color = "#A23B72", size = 2, alpha = 0.6) +
  labs(
    title = "Time Series: Monthly US Electricity Generation",
    x = "Date",
    y = "Generation (GWh)",
    subtitle = paste("Data range:", min(plot_data$date), "to", max(plot_data$date))
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.major = element_line(color = "gray90")
  )

print(p1)
ggsave("time_series_plot.png", p1, width = 12, height = 6)
cat("✓ Saved: time_series_plot.png\n")

# Plot 2: Distribution of Values
p2 <- ggplot(plot_data, aes(x = value)) +
  geom_histogram(bins = 20, fill = "#2E86AB", alpha = 0.7, color = "black") +
  geom_vline(aes(xintercept = mean(value, na.rm = TRUE)),
             color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(aes(xintercept = median(value, na.rm = TRUE)),
             color = "green", linetype = "dotted", linewidth = 1) +
  labs(
    title = "Distribution of Generation Values",
    x = "Generation (GWh)",
    y = "Frequency",
    subtitle = paste("Mean:", round(mean(plot_data$value, na.rm = TRUE), 2),
                     "| Median:", round(median(plot_data$value, na.rm = TRUE), 2))
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

print(p2)
ggsave("distribution_plot.png", p2, width = 10, height = 6)
cat("✓ Saved: distribution_plot.png\n")

# Plot 3: ACF Plot (Autocorrelation)
png("acf_plot.png", width = 800, height = 600)
acf(plot_data$value, main = "Autocorrelation Function (ACF)",
    xlab = "Lag (months)", ylab = "ACF")
dev.off()
cat("✓ Saved: acf_plot.png\n")

# Plot 4: Trend decomposition
# Simple decomposition using smooth
trend_smooth <- zoo::rollmean(plot_data$value, k = 12, fill = NA)
detrended <- plot_data$value - trend_smooth

p4 <- ggplot(data.frame(date = plot_data$date, trend = trend_smooth),
             aes(x = date, y = trend)) +
  geom_line(color = "#E63946", linewidth = 1) +
  labs(
    title = "Trend Component (12-month Moving Average)",
    x = "Date",
    y = "Trend (GWh)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

print(p4)
ggsave("trend_plot.png", p4, width = 12, height = 6)
cat("✓ Saved: trend_plot.png\n")

# ============================================================================
# STEP 9: Summary Statistics
# ============================================================================
cat("\n========== DATA SUMMARY REPORT ==========\n")
cat("Sample: Monthly US Electricity Generation (2020-2024)\n")
cat("Observations:", nrow(plot_data), "\n")
cat("Mean:", round(mean(plot_data$value), 2), "GWh\n")
cat("Median:", round(median(plot_data$value), 2), "GWh\n")
cat("Std Dev:", round(sd(plot_data$value), 2), "GWh\n")
cat("Min:", round(min(plot_data$value), 2), "GWh\n")
cat("Max:", round(max(plot_data$value), 2), "GWh\n")
cat("Range:", round(max(plot_data$value) - min(plot_data$value), 2), "GWh\n\n")

# Save cleaned data
saveRDS(dt_clean, "cleaned_data.rds")
cat("✓ Cleaned data saved to: cleaned_data.rds\n\n")

cat("========================================\n")
cat("Data is ready for time series forecasting!\n")
cat("========================================\n\n")

cat("Next steps:\n")
cat("1. ✓ Data exploration (COMPLETED)\n")
cat("2. Stationarity testing (ADF test)\n")
cat("3. Seasonal decomposition\n")
cat("4. Train/test split\n")
cat("5. Model building (ARIMA, ETS, Prophet)\n")
cat("6. Forecasting and evaluation\n\n")
