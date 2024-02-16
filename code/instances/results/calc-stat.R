## Code for classification of nondominated points and updating statistics

library(tidyverse)
library(gMOIP)
here::i_am("code/instances/results/calc-stat.R")  # specify relative path given project

#### Functions ####
#' Classify and update statistics
#'
#' @param path Path to result file.
#' @return True if calculated statistics.
calcStat <- function(path) {
   tictoc::tic()
   cat("Check:", path, "...")
   lst <- jsonlite::read_json(path, simplifyVector = T)
   calc <- any(is.na(lst$points$cls))
   if (calc) {
      p <- lst$statistics$p
      pts <- classifyNDSet(lst$points[, 1:p])
      pts <- pts %>% distinct()
      lst$points <- pts %>% select(-se, -sne, -us)
      lst$statistics$card <- nrow(pts)
      lst$statistics$supported <- sum(pts$se) + sum(pts$sne)
      lst$statistics$extreme <- sum(pts$se)
      lst$statistics$unsupported <- sum(pts$us)
      lst$statistics$min <- Rfast::colMins(as.matrix(pts[, 1:p]), value = T)
      lst$statistics$max <- Rfast::colMaxs(as.matrix(pts[, 1:p]), value = T)
      lst$statistics$width <- Rfast::colrange(as.matrix(pts[, 1:p]))
      lst$statistics$method <- NULL
      jsonlite::write_json(lst, path, pretty = TRUE)
      cat(" done.\n")
   } else {
      cat(" already calc.\n")
   }
   lst <- tictoc::toc()
   return(calc)
}


#### Run script ####
## Open log file
zz <- file(here::here("code/instances/results/calc-stat.log"), open = "wt")
sink(zz, type = "output", split = T)   # open the file for output
sink(zz, type = "message")  # open the same file for messages, errors and warnings

paths <- fs::dir_ls(here::here("code/instances/results"), recurse = T, type = "file", glob = "*prob*.json")
timeLimit <- 1 * 60  # max run time in sec
tictoc::tic.clear()
start <- Sys.time()
for (path in paths) {
   calcStat(path)
   cpu <- Sys.time() - start 
   cat("Cpu test", cpu, "\n")
   if (cpu > timeLimit) {
      message("Time limit reached! Stop R script.")
      break
   }
}
cat("\n\nFinish running R script.\n\n")

## Close log file
sink(type = "message")  # close the file for output
sink()  # close the file for messages, errors and warnings


#### Tests ####
# path <- paths[1]
# calcStat(path)
# 
# pts[duplicated(pts),]
# pts$nd <- T
# plotly::plot_ly(pts, x = ~z1, y = ~z2)
