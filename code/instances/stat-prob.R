## Code for classification of nondominated points and updating statistics

remotes::install_github("relund/gMOIP")
library(tidyverse)
library(gMOIP)
here::i_am("code/instances/stat-prob.R")  # specify relative path given project

#### Functions ####
#' Classify and update statistics
#'
#' @param path Path to result file.
#' @return True if calculated statistics.
calcStat <- function(path) {
   tictoc::tic()
   cat("Update statistics:", path, "...")
   lst <- jsonlite::read_json(path, simplifyVector = T)
   calc <- any(is.na(lst$statistics$min))
   if (calc) {
      p <- lst$statistics$p
      pts <- lst$points
      lst$statistics$min <- Rfast::colMins(as.matrix(pts[, 1:p]), value = T)
      lst$statistics$max <- Rfast::colMaxs(as.matrix(pts[, 1:p]), value = T)
      lst$statistics$width <- Rfast::colrange(as.matrix(pts[, 1:p]))
      lst$statistics$method <- NULL
      jsonlite::write_json(lst, path, pretty = TRUE)
      cat(" done.\n")
   } else {
      cat(" already calc.\n")
   }
   tictoc::toc()
   return(calc)
}

classifyStat <- function(path) {
   tictoc::tic()
   lst <- jsonlite::read_json(path, simplifyVector = T)
   cat("Classify", lst$statistics$card, "points:", path, "...")
   calc <- is.null(lst$points$cls) | any(is.na(lst$points$cls))
   if (calc) {
      p <- lst$statistics$p
      pts <- classifyNDSet(lst$points[, 1:p])
      pts <- pts %>% distinct()
      lst$points <- pts %>% select(-se, -sne, -us)
      lst$statistics$card <- nrow(pts)
      lst$statistics$supported <- sum(pts$se) + sum(pts$sne)
      lst$statistics$extreme <- sum(pts$se)
      lst$statistics$unsupported <- sum(pts$us)
      jsonlite::write_json(lst, path, pretty = TRUE)
      cat(" done.\n")
   } else {
      cat(" already calc.\n")
   }
   tictoc::toc()
   return(calc)
}

updateProbStatFile <- function() {
   cat("Update statistics for results.")
   paths <- fs::dir_ls(here::here("code/instances/results"), recurse = T, type = "file", glob = "*prob*.json")
   prefix <- str_extract(paths, ".*/")
   filename <- str_extract(paths, "^.*/(.*)$", group = 1)
   alg <- unique(str_extract(filename, "(.*?)-", group = 1))
   for (a in alg) {
      if (a == "alg1") {
         datSubProb <- read_csv(here::here("code/instances/stat-sp.csv"), show_col_types = F)
         datRes <- NULL
         algPaths <- str_subset(paths, "alg1")
         algFiles <- str_extract(algPaths, "^.*/instances/(.*)$", group = 1)
         probFiles <- str_extract(algFiles, "^.*alg1-(.*)$", group = 1)
         for (i in 1:length(algPaths)) {
            lstAlg <- jsonlite::read_json(algPaths[i])
            row <- c(path = algFiles[i], unlist(lstAlg$statistics))
            subProb <- unlist(jsonlite::read_json(here::here("code/instances/problems", probFiles[i])))
            dat <- datSubProb %>% filter(path %in% subProb)
            row <- c(row, spCard = dat$card, spSupported = dat$supported, spExtreme = dat$extreme, spUnsupported = dat$unsupported)
            datRes <- datRes %>% bind_rows(row)
         }
         write_csv(datRes, here::here("code/instances/stat-prob.csv"))
      }
   }
   return(alg)
}


#### Run script ####
## Open log file
zz <- file(here::here("code/instances/results/calc-stat.log"), open = "wt")
sink(zz, type = "output", split = T)   # open the file for output
# sink(zz, type = "message")  # open the same file for messages, errors and warnings

paths <- fs::dir_ls(here::here("code/instances/results"), recurse = T, type = "file", glob = "*prob*.json")
timeLimit <- 10 * 60  # max run time in sec
tictoc::tic.clear()
start <- Sys.time()
for (path in paths) {
   calcStat(path)
   cpu <- difftime(Sys.time(), start, units = "secs")
   cat("Cpu total", cpu, "\n")
   if (cpu > timeLimit) {
      message("Time limit reached! Stop R script.")
      break
   }
}
updateProbStatFile()
if (cpu < timeLimit) {
   for (path in paths) {
      classifyStat(path)
      cpu <- difftime(Sys.time(), start, units = "secs")
      cat("Cpu total", cpu, "\n")
      if (cpu > timeLimit) {
         message("Time limit reached! Stop R script.")
         break
      }
   }
   updateProbStatFile()
} 

cat("\n\nFinish running R script.\n\n")

## Close log file
# sink(type = "message")  # close the file for output
sink()  # close the file for messages, errors and warnings


#### Tests ####
# path <- paths[1]
# calcStat(path)
# 
# pts[duplicated(pts),]
# pts$nd <- T
# plotly::plot_ly(pts, x = ~z1, y = ~z2)
