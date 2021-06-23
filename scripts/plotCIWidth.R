rm(list=ls())
setwd('../s2-fewshot/')

df = read.csv(file='episodes_vs_test_examples_1000_noise_05.csv', header=T, stringsAsFactors=F)
df$budget_hours = factor(df$budget_hours, levels=rev(unique(df$budget_hours)))

library(ggplot2)
library(dplyr)


# remotes::install_github("slowkow/ggrepel")
# https://github.com/slowkow/ggrepel/issues/184
library(ggrepel)


# this only works for CI correctness, copy pasta for other metrics
computePolygonForBudget = function(df, budget, lower_prob = 0.1, upper_prob = 0.9){
  sub_df = subset(df, budget_hours == budget)
  upper = group_by(sub_df, n_episodes) %>% summarize(ci_width=quantile(ci_width, probs=upper_prob))
  lower = group_by(sub_df, n_episodes) %>% summarize(ci_width=quantile(ci_width, probs=lower_prob))
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
df_avg_ci_width = dplyr::group_by(df, n_episodes, n_test_examples, budget_hours) %>% summarize(ci_width=mean(ci_width))
df_avg_ci_width$budget_hours = as.factor(df_avg_ci_width$budget_hours)
df_avg_ci_width$n_test_examples = plyr::round_any(df_avg_ci_width$n_test_examples * 4.66, 10)

library(viridis)
COLOR_SCHEME = 'rocket'
ggplot() + 
  geom_polygon(data=poly_df, aes(x=n_episodes, y=ci_width, group=budget_hours, fill=budget_hours), alpha=0.3) +
  geom_line(data=df_avg_ci_width, aes(x=n_episodes, y=ci_width, group=budget_hours, color=budget_hours), lwd=2, alpha=1.0) +
  # scale_color_viridis_d(name='Budget hours', option=COLOR_SCHEME) +
  # scale_fill_viridis_d(name='Budget hours', option=COLOR_SCHEME) +
  scale_color_brewer(name='Budget hours', palette = "Dark2") +
  scale_fill_brewer(name='Budget hours', palette = "Dark2") +
  xlab('Number of episodes') +
  ylab('Confidence interval width') +
  scale_x_continuous(breaks=c(5, seq(15, 150, 15))) +
  scale_y_continuous(breaks=seq(0.02, 0.08, 0.01)) +
  theme(legend.position="top", text=element_text(size = 30)) +
  guides(color=guide_legend(ncol=3, reverse=TRUE), fill=guide_legend(ncol=3, reverse=TRUE)) +
  geom_label_repel(data=df_avg_ci_width, aes(x=n_episodes, y=ci_width, label=n_test_examples, group=budget_hours, color=budget_hours), size=5.5, show.legend = F, max.overlaps=5)



# This is just to compute the relative marginal improvements beyond 36 hr, which is the first viable budget
narrowest_df = NULL
for(budget in unique(df_avg_ci_width$budget_hours)){
  df_sub = subset(df_avg_ci_width, budget_hours==budget & n_episodes >= 60)
  narrowest_row = df_sub[order(df_sub$ci_width), ][1,]
  narrowest_df = rbind(narrowest_df, narrowest_row)
}
narrowest_df$diff = c(abs(diff(narrowest_df$ci_width)), 0)
narrowest_df$rel_diff = narrowest_df$diff / narrowest_df$ci_width
narrowest_df$rel_diff_36hr = (narrowest_df$ci_width - subset(narrowest_df, budget_hours==36)$ci_width) / subset(narrowest_df, budget_hours==36)$ci_width

