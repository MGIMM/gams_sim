library(rjson)

json_file <- '../py/json/original_multinomial_n_rep_100_k_50_n_sim_50000_a_0.1_b_4.9_beta_1.0_dt_0.01_mu_1.0.json'
json_data <- fromJSON(file=json_file)


V_list <- json_data["V_list"][[1]]
E_list <- json_data["E_list"][[1]]

sample_start = 10
sample_size = 40000

V_list_sample <- V_list[sample_start:(sample_start+sample_size-1)]
E_list_sample <- E_list[sample_start:(sample_start+sample_size-1)]

v_mean <- cumsum(V_list[1:(sample_start+sample_size-1)])/1:(sample_start+sample_size-1)
v_mean <- v_mean[sample_start:(sample_start+sample_size-1)]

v_std <- sapply(1:sample_size, function(k) {return(sd(V_list_sample[1:k]))})


e <- qnorm(0.975)*v_std
e <- e/sample_start:(sample_start+sample_size-1)

v_naive <- sapply(1:(sample_start+sample_size-1), function(k) {return(var(E_list[1:k]))})
v_naive <- v_naive[sample_start:(sample_start+sample_size-1)]

df = data.frame(v_mean, v_mean-e, v_mean +e, v_naive, sample_start:(sample_start+sample_size-1))
colnames(df) = c('v_mean','ci_lower','ci_upper', 'v_naive','n_sim')

library(ggplot2)
theme_set(theme_bw())
theme_update(panel.background = element_rect(fill = 'white', colour = "white")
       ,panel.grid.major = element_line(colour = 'grey80',linetype='dotted')
       ,panel.grid.minor = element_line(colour = 'grey80',linetype='dotted')
       ,legend.box.background = element_rect(colour = 'black'))

ggplot(data= df,aes(x = n_sim))+
  geom_line(aes(y = v_mean, colour = 'mean of variance estimator',
                linetype = 'mean of variance estimator'),
  )+
  geom_line(aes(y = ci_lower, colour = 'lower bound of 95% CI',
                linetype = 'lower bound of 95% CI'),
  )+
  geom_line(aes(y = ci_upper, colour = 'upper bound of 95% CI',
                linetype = 'upper bound of 95% CI'),
  )+
  geom_line(aes(y = v_naive, colour = 'naive variance estimator',
                linetype = 'naive variance estimator'),
  )+
  xlab('number of simulations')+
  ylab('values')+
  scale_colour_manual('legend',breaks = c('mean of variance estimator',
                                          'lower bound of 95% CI',
                                          'upper bound of 95% CI',
                                          'naive variance estimator'),
                        values = c('mean of variance estimator'='black',
                                   'lower bound of 95% CI'='darkred' ,
                                   'upper bound of 95% CI'='darkred',
                                   'naive variance estimator'='darkgreen') )+
  scale_linetype_manual('legend',breaks = c('mean of variance estimator',
                                          'lower bound of 95% CI',
                                          'upper bound of 95% CI',
                                          'naive variance estimator'),
                        values = c('mean of variance estimator'='solid',
                                   'lower bound of 95% CI'='dotted' ,
                                   'upper bound of 95% CI'='dotted',
                                   'naive variance estimator'='longdash') )

  