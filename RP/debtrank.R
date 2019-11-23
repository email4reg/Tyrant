library(Matrix)
library(NetworkRiskMeasures)

library(ggplot2)
library(ggnetwork)
library(igraph)
library(intergraph)

# clear
rm(list = ls())

# data("sim_data")
# head(sim_data)
data <- read.csv('/Users/hehaoran/Desktop/data/bank_specific_date_(2007, 3, 31).csv')

set.seed(15)


# md_mat <- matrix_estimation(
  # sim_data$assets, sim_data$liabilities, method = "md", verbose = FALSE)

md_mat <- matrix_estimation(
  data$inter_bank_assets, data$inter_bank_liabilities, method = "md", verbose = FALSE)
# rownames and colnames for the matrix
rownames(md_mat) <- colnames(md_mat) <- data$bank_name

## converting our network to an igraph object
gmd <- graph_from_adjacency_matrix(md_mat, weighted = T)

# adding other node attributes to the network
V(gmd)$buffer <- sim_data$buffer
V(gmd)$weights <- sim_data$weights/sum(sim_data$weights)
V(gmd)$assets  <- sim_data$assets
V(gmd)$liabilities <- sim_data$liabilities

## ploting with ggplot and ggnetwork
set.seed(20)
netdf <- ggnetwork(gmd)

ggplot(netdf, aes(x = x, y = y, xend = xend, yend = yend)) + 
  geom_edges(arrow = arrow(length = unit(6, "pt"), type = "closed"), 
             color = "grey50", curvature = 0.1, alpha = 0.5) + 
  geom_nodes(aes(size = weights)) + 
  ggtitle("Estimated interbank network") + 
  theme_blank()

# network density
edge_density(gmd)

# assortativity
assortativity_degree(gmd)

## Finding central, important or systemic nodes on the network
sim_data$degree <- igraph::degree(gmd)
sim_data$btw    <- igraph::betweenness(gmd)
sim_data$close  <- igraph::closeness(gmd)
sim_data$eigen  <- igraph::eigen_centrality(gmd)$vector
sim_data$alpha  <- igraph::alpha_centrality(gmd, alpha = 0.5)

sim_data$imps <- impact_susceptibility(exposures = gmd, buffer = sim_data$buffer)
sim_data$impd <- impact_diffusion(
  exposures = gmd, buffer = sim_data$buffer, weights = sim_data$weights)$total

## Contagion metrics: default cascades and DebtRank
# DebtRank simulation
contdr <- contagion(exposures = md_mat, buffer = sim_data$buffer, weights = sim_data$weights, 
                     shock = "all", method = "debtrank", verbose = FALSE)
summary(contdr)
plot(contdr)

# interpret these results
contdr_summary <- summary(contdr)
sim_data$DebtRank <- contdr_summary$summary_table$additional_stress


## Traditional default cascades simulation
contthr <-  contagion(exposures = md_mat, buffer = sim_data$buffer, weights = sim_data$weights, 
                       shock = "all", method = "threshold", verbose = FALSE)
summary(contthr)

contthr_summary <- summary(contthr)
sim_data$cascade <- contthr_summary$summary_table$additional_stress

## 
rankings <- sim_data[1]
rankings <- cbind(rankings, lapply(sim_data[c("DebtRank","cascade","degree","eigen","impd","assets", "liabilities", "buffer")], 
                                   function(x) as.numeric(factor(-1*x))))
rankings <- rankings[order(rankings$DebtRank), ]
head(rankings, 10)

cor(rankings[-1])

# simulating shock scenarios 1% to 25% shock in all vertices
s <- seq(0.01, 0.25, by = 0.01)
shocks <- lapply(s, function(x) rep(x, nrow(md_mat)))
names(shocks) <- paste(s*100, "pct shock")

cont <- contagion(exposures = md_mat, buffer = sim_data$buffer, shock = shocks, 
                  weights = sim_data$weights, method = "debtrank", verbose = FALSE)
summary(cont)
plot(cont)

