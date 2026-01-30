library(tidyverse)
library(skimr)
library(janitor)

df <- read.csv("/Users/sangmin/Desktop/KW/kwco/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df <- df %>%
  mutate(TotalCharges = as.numeric(TotalCharges)) %>%
  drop_na(TotalCharges)

skim(df)
df %>% tabyl(Churn)
df %>% tabyl(Contract, Churn) %>% adorn_percentages("row") %>% adorn_pct_formatting()