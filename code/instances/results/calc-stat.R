## Code for classification of nondominated points and updating statistics


library(tidyverse)
library(gMOIP)
here::i_am("code/instances/results/calc-stat.R")  # specify relative path given project

#### Functions ####
#' Classify and update statistics
#'
#' @param path Path to result file.
#' @return The path if modified the file; otherwise `NULL`.
calcStat <- function(path) {
   res <- NULL
   lst <- jsonlite::read_json(path, simplifyVector = T)
   if (any(is.na(lst$points$cls))) {
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
      res <- path
   }
   return(invisible(res))
}


#### Run script ####
## Open log file
zz <- file(here::here("code/instances/results/calc-stat.log"), open = "wt")
sink(zz, type = "output")   # open the file for output
sink(zz, type = "message")  # open the same file for messages, errors and warnings

paths <- fs::dir_ls(here::here("code/instances/results"), recurse = T, type = "file", glob = "*prob*.json")[1:6]
lst <- map(paths, calcStat)
res <- unlist(lst)
names(res) <- NULL
cat("Files updated:\n")
res

## Close log file
sink()  # close the file for output
sink()  # close the file for messages, errors and warnings


#### Tests ####
# path <- paths[1]
# calcStat(path)
# 
# pts[duplicated(pts),]
# pts$nd <- T
# plotly::plot_ly(pts, x = ~z1, y = ~z2)
