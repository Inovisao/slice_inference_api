library("ggplot2")
library("gridExtra")
library("plyr")
library("scales")
library("dplyr")
library(tidyr)
library("Metrics")
library(data.table)

options(scipen = 999)

# Paleta de cores fixa por modelo (consistente em todos os gráficos)
model_colors <- c(
  "Baseline_YOLOv8" = "#1A1A2E",
  "Baseline_Faster" = "#8B4513",
  "Baseline_Detr"   = "#8B0000",
  "SAHI_YOLOv8"     = "#4E79A7",
  "SAHI_Faster"     = "#F28E2B",
  "SAHI_Detr"       = "#E15759",
  "ASAHI_YOLOv8"    = "#76B7B2",
  "ASAHI_Faster"    = "#59A14F",
  "ASAHI_Detr"      = "#B07AA1"
)

# -------------------------------------------------------------------
# BOXPLOT — um arquivo por métrica em results/plots/boxplots/
#
dados_slicing <- read.table('results/results.csv', sep = ',', header = TRUE)

dados_baseline <- read.table('results/baseline/results.csv', sep = ',', header = TRUE)
baseline_name_map <- c("YOLOV8" = "Baseline_YOLOv8", "Faster" = "Baseline_Faster", "Detr" = "Baseline_Detr")
dados_baseline$models <- baseline_name_map[dados_baseline$ml]
dados_baseline$fold   <- as.integer(sub("fold_", "", dados_baseline$fold))
dados_baseline <- dados_baseline[, c("models", "fold", "mAP50", "mAP75", "mAP",
                                      "precision", "recall", "fscore", "MAE", "RMSE", "r")]

dados <- rbind(
  dados_slicing[, c("models", "fold", "mAP50", "mAP75", "mAP",
                    "precision", "recall", "fscore", "MAE", "RMSE", "r")],
  dados_baseline
)
dados$models <- factor(dados$models, levels = names(model_colors))

dir.create("results/plots/boxplots", recursive = TRUE, showWarnings = FALSE)
dir.create("results/plots/counting",  recursive = TRUE, showWarnings = FALSE)

metricas_label <- list(
  mAP50     = "mAP@50",
  mAP75     = "mAP@75",
  mAP       = "mAP@[.5:.95]",
  precision = "Precision",
  recall    = "Recall",
  fscore    = "F1-Score",
  MAE       = "MAE",
  RMSE      = "RMSE",
  r         = "Pearson r"
)

for (metrica in names(metricas_label)) {
  cat("Gerando boxplot:", metrica, "\n")
  ylabel <- metricas_label[[metrica]]

  g <- ggplot(dados, aes(x = models, y = .data[[metrica]], fill = models)) +
       geom_boxplot(width = 0.6, outlier.shape = 21, outlier.size = 2) +
       geom_jitter(width = 0.12, size = 1.5, alpha = 0.7, color = "black") +
       scale_fill_manual(values = model_colors) +
       labs(title = ylabel, x = NULL, y = ylabel) +
       theme_bw(base_size = 13) +
       theme(
         legend.position  = "none",
         plot.title       = element_text(hjust = 0.5, face = "bold"),
         axis.text.x      = element_text(angle = 35, hjust = 1, size = 11),
         panel.grid.minor = element_blank()
       )

  ggsave(sprintf("results/plots/boxplots/%s.png", metrica), g,
         width = 7, height = 5, dpi = 150)
}

# Grid com todos juntos (visão geral)
graficos <- lapply(names(metricas_label), function(metrica) {
  ggplot(dados, aes(x = models, y = .data[[metrica]], fill = models)) +
    geom_boxplot(width = 0.6, outlier.shape = 21, outlier.size = 1.2) +
    scale_fill_manual(values = model_colors) +
    labs(title = metricas_label[[metrica]], x = NULL, y = NULL) +
    theme_bw(base_size = 9) +
    theme(
      legend.position  = "none",
      plot.title       = element_text(hjust = 0.5, face = "bold", size = 9),
      axis.text.x      = element_text(angle = 40, hjust = 1, size = 7),
      panel.grid.minor = element_blank()
    )
})
g_all <- grid.arrange(grobs = graficos, ncol = 3)
ggsave("results/plots/boxplots/_overview.png", g_all, width = 14, height = 11, dpi = 150)

# -------------------------------------------------------------------
# XY CONTAGEM MANUAL X AUTOMÁTICA - JUNTANDO TODAS AS DOBRAS
#
dadosContagem <- read.table('results/counting.csv', sep = ',', header = TRUE)
dadosContagem$models <- factor(dadosContagem$models, levels = names(model_colors))

nets <- levels(droplevels(dadosContagem[!is.na(dadosContagem$models), ]))
nets <- as.character(unique(dadosContagem$models[!is.na(dadosContagem$models)]))
cat("Modelos encontrados:", paste(nets, collapse = ", "), "\n")

graficos_count <- list()
for (net in nets) {
  filtrado <- dadosContagem[dadosContagem$models == net, ]

  RMSE_val = rmse(filtrado$groundtruth, filtrado$predicted)
  MAE_val  = mae(filtrado$groundtruth, filtrado$predicted)
  MAPE_val = mape(filtrado$groundtruth, filtrado$predicted)
  R_val    = cor(filtrado$groundtruth, filtrado$predicted, method = "pearson")
  TITULO   = sprintf("%s\nRMSE=%.3f  MAE=%.3f  r=%.3f", net, RMSE_val, MAE_val, R_val)
  MAX      = max(filtrado$groundtruth, filtrado$predicted) * 1.05

  cor_modelo <- model_colors[net]

  g <- ggplot(filtrado, aes(x = groundtruth, y = predicted)) +
       geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
       geom_point(color = cor_modelo, size = 2.5, alpha = 0.8) +
       geom_smooth(method = 'lm', se = TRUE, color = cor_modelo, fill = cor_modelo, alpha = 0.15) +
       labs(title = TITULO,
            x = "Contagem Manual (GT)", y = "Contagem Predita") +
       theme_bw(base_size = 11) +
       theme(plot.title = element_text(size = 9, hjust = 0.5)) +
       xlim(0, MAX) + ylim(0, MAX)

  ggsave(sprintf("results/plots/counting/%s.png", net), g,
         width = 5, height = 5, dpi = 150)
  graficos_count[[net]] <- g
}

g_count_all <- grid.arrange(grobs = graficos_count, ncol = 2)
ggsave("results/plots/counting/_overview.png", g_count_all, width = 10, height = 14, dpi = 150)

# -------------------------------------------------------------------
# ANOVA + Tukey HSD para cada métrica
#
output_file <- 'results/anova_all_results.txt'
sink(output_file); sink()

if (!require(multcomp)) install.packages("multcomp", dependencies = TRUE)
if (!require(multcompView)) install.packages("multcompView", dependencies = TRUE)
library(multcomp)
library(multcompView)

realizar_anova <- function(df, metric, output_file) {
  anova_result <- tryCatch({
    aov(as.formula(paste(metric, "~ models")), data = df)
  }, error = function(e) {
    message("Erro ao realizar ANOVA para ", metric, ": ", e)
    return(NULL)
  })

  if (!is.null(anova_result)) {
    anova_summary <- summary(anova_result)

    sink(output_file, append = TRUE)
    cat("\n------------------------------------------------------------\n")
    cat("ANOVA para", metric, "\n")
    print(anova_summary)

    if ("Pr(>F)" %in% colnames(anova_summary[[1]]) && anova_summary[[1]][["Pr(>F)"]][1] < 0.05) {
      tukey_result <- TukeyHSD(anova_result)
      cat("\nTukey HSD para", metric, "\n")
      print(tukey_result)

      tukey_df <- as.data.frame(tukey_result[[1]])

      cld_result <- cld(glht(anova_result, linfct = mcp(models = "Tukey")))
      cat("\nCLD para", metric, "\n")
      print(cld_result)
    }

    sink()
  } else {
    message("ANOVA não pôde ser realizada para ", metric)
  }
}

metrics <- c("mAP", "mAP50", "mAP75", "MAE", "RMSE", "r", "precision", "recall", "fscore")
for (metric in metrics) {
  realizar_anova(dados, metric, output_file)
}

cat("Os resultados foram salvos em results/\n")
