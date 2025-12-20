# ===============================
# DLD vs TD: Logistic Regression with GLM
# ===============================

library(here)

# Load data
data <- read.csv(here("data_for_R", "ml_sli_data_for_R_4_6_all.csv"))

# Inspect data
head(data)

# -------------------------------
# Standardize continuous predictors (z-score) prior to model fitting
# -------------------------------
data$MLU_with_mazes <- as.numeric(scale(data$MLU_with_mazes)) #numeric is optional here
data$TNU <- as.numeric(scale(data$TNU))
data$Types_with_mazes <- as.numeric(scale(data$Types_with_mazes))
data$FCWR <- as.numeric(scale(data$FCWR))
data$Verb_Ratio <- as.numeric(scale(data$Verb_Ratio))
data$Noun_Ratio <- as.numeric(scale(data$Noun_Ratio))
data$Age <- as.numeric(scale(data$Age))  # optional for Age standardized

# -------------------------------
# Logistic regression: Narrative Microstructure
# -------------------------------
glm_nm <- glm(
  label ~ FCWR + Verb_Ratio + Noun_Ratio, family = binomial, data = data)
summary(glm_nm)

# -------------------------------
# Logistic regression: interaction model
# -------------------------------
glm_age <- glm(
  label ~ Age * FCWR,
  family = binomial,
  data = data
)
summary(glm_age)
