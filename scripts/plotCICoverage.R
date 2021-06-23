rm(list=ls())
setwd('../s2-fewshot/')

df = read.csv(file='episodes_vs_test_examples_1000_noise_05.csv', header=T, stringsAsFactors=F)
df$budget_hours = factor(df$budget_hours, levels=rev(unique(df$budget_hours)))

library(ggplot2)
library(dplyr)


# this only works for CI coverage, copy pasta for other metrics
computePolygonForBudget = function(df, budget, lower_prob = 0.10, upper_prob = 0.90){
  sub_df = subset(df, budget_hours == budget)
  upper = group_by(sub_df, n_episodes) %>% summarize(ci_coverage=quantile(ci_coverage, probs=upper_prob))
  lower = group_by(sub_df, n_episodes) %>% summarize(ci_coverage=quantile(ci_coverage, probs=lower_prob))
  combined = rbind(upper, lower[nrow(lower):1,])
  return(combined)
}

# compute error bars
poly_df = NULL
for(budget in unique(df$budget_hours)){
  poly_for_budget = computePolygonForBudget(df=df, budget=budget)
  poly_for_budget$budget_hours = budget
  poly_df = rbind(poly_df, poly_for_budget)
}
poly_df$budget_hours = factor(poly_df$budget_hours, levels=rev(unique(poly_df$budget_hours)))

# compute mean
df_avg_ci_coverage = dplyr::group_by(df, n_episodes, budget_hours) %>% summarize(mean_ci_coverage=median(ci_coverage))
df_avg_ci_coverage$budget_hours = as.factor(df_avg_ci_coverage$budget_hours)


library(viridis)
COLOR_SCHEME = 'magma'
ggplot() + 
  geom_polygon(data=poly_df, aes(x=n_episodes, y=ci_coverage, group=budget_hours, fill=budget_hours), alpha=0.25) +
  geom_line(data=df_avg_ci_coverage, aes(x=n_episodes, y=mean_ci_coverage, group=budget_hours, color=budget_hours), lwd=2, alpha=1) +
  scale_color_brewer(name='Budget hours', palette = "Dark2") +
  scale_fill_brewer(name='Budget hours', palette = "Dark2") +
  geom_hline(yintercept=0.95, lwd=1, color='black', linetype='dashed') +
  xlab('Number of episodes') +
  ylab('Coverage probability') +
  scale_x_continuous(breaks=c(5, seq(15, 150, 15))) +
  scale_y_continuous(breaks=seq(0.82, 0.96, 0.02)) +
  theme(legend.position="top", text=element_text(size = 30)) +
  guides(color=guide_legend(ncol=3, reverse=TRUE), fill=guide_legend(ncol=3, reverse=TRUE))





