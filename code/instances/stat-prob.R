## Code for classification of nondominated points and updating statistics

remotes::install_github("relund/gMOIP")
library(tidyverse)
library(gMOIP)
library(tryCatchLog)
here::i_am("code/instances/stat-prob.R")  # specify relative path given project
options(readr.show_progress = FALSE)

#### Functions ####

#' Update simple statistics
#'
#' @param path Path to result file.
#' @return True if calculated statistics.
calcStat <- function(path) {
   tictoc::tic()
   cat("Update simple statistics:", path, "...")
   lst <- jsonlite::read_json(path, simplifyVector = T)
   calc <- any(is.na(lst$statistics$min)) & length(lst$points) > 0
   if (calc) {
      p <- lst$statistics$p
      pts <- lst$points
      lst$statistics$min <- Rfast::colMins(as.matrix(pts[, 1:p]), value = T)
      lst$statistics$max <- Rfast::colMaxs(as.matrix(pts[, 1:p]), value = T)
      lst$statistics$width <- Rfast::colrange(as.matrix(pts[, 1:p]))
      # lst$statistics$method <- NULL
      jsonlite::write_json(lst, path, pretty = FALSE)
      cat(" done.\n")
      datOkay <- bind_rows(datOkay, c(path = path, type = "simple", alg = "alg1"))
      write_csv(datOkay, file = here::here("code/instances/stat-prob-okay.csv"))
   } else {
      datOkay <<- bind_rows(datOkay, c(path = path, type = "simple", alg = "alg1"))
      write_csv(datOkay, file = here::here("code/instances/stat-prob-okay.csv"))
      cat(" already calc.\n")
   }
   tictoc::toc()
   return(calc)
}



#' Classify points into se, sne and us
#'
#' @param path Path to result file.
#' @param classifyExt If true only classify se.
#'
#' @return True if classified any points
classifyStat <- function(path, classifyExt = FALSE) {
   cat(path, ": Classify", lst$statistics$card, "points ...")
   tictoc::tic()
   lst <- jsonlite::read_json(path, simplifyVector = T)
   if (is.null(lst$points)) {
      calc <- FALSE
   } else {
      calc <- is.null(lst$points$cls) #| any(is.na(lst$points$cls))
   }
   if (!calc & !is.null(lst$points$cls) & any(is.na(lst$points$cls))) {
      datOkay <<- bind_rows(datOkay, c(path = path, type = "classifyExt", alg = "alg1"))
      write_csv(datOkay, file = here::here("code/instances/stat-prob-okay.csv"))
   } 
   if (!calc & !is.null(lst$points$cls) & !any(is.na(lst$points$cls)))  {
      datOkay <<- bind_rows(datOkay, c(path = path, type = "classify", alg = "alg1"))
      write_csv(datOkay, file = here::here("code/instances/stat-prob-okay.csv"))
   }
   if (calc & !classifyExt) {
      p <- lst$statistics$p
      pts <- classifyNDSet(lst$points[, 1:p]);
      if (nrow(pts) > 0) pts <- pts %>% distinct();
      lst$points <- pts %>% select(-se, -sne, -us);
      lst$statistics$card <- nrow(pts);
      lst$statistics$supported <- sum(pts$se) + sum(pts$sne);
      lst$statistics$extreme <- sum(pts$se);
      lst$statistics$unsupported <- sum(pts$us);
      jsonlite::write_json(lst, path, pretty = FALSE)
      cat(" done.\n")
      datOkay <<- bind_rows(datOkay, c(path = path, type = "classify", alg = "alg1"))
      write_csv(datOkay, file = here::here("code/instances/stat-prob-okay.csv"))
   } else if (calc & classifyExt) {
      cat(path, ": Classify", lst$statistics$card, " points (find extreme) ...")
      p <- lst$statistics$p
      pts <- classifyExt(lst$points[, 1:p]);
      if (nrow(pts) > 0) pts <- pts %>% distinct();
      lst$points <- pts %>% select(-se);
      lst$statistics$card <- nrow(pts);
      lst$statistics$extreme <- sum(pts$se);
      jsonlite::write_json(lst, path, pretty = FALSE)
      cat(" done (ext).\n")
      datOkay <<- bind_rows(datOkay, c(path = path, type = "classifyExt", alg = "alg1"))
      write_csv(datOkay, file = here::here("code/instances/stat-prob-okay.csv"))
   } else {
      cat(" already calc.\n")
   }
   if (fs::file_size(path) > "20MB" & !any(is.na(lst$statistics$extreme))) { # reduce file sizes
      lst$points <- NULL
      jsonlite::write_json(lst, path, pretty = FALSE)
   }
   tictoc::toc()
   return(calc)
}


#' Classify points into exteme and non-extreme
#'
#' @param pts Nondominted set
#'
#' @return The nondominted set with extra columns cls (chr) and se (0 or 1).
classifyExt <- function(pts) {
   p <- ncol(pts)
   colnames(pts)[1:p] <- paste0("z", 1:p)
   direction <- rep(1,p)
   nadir <-
      purrr::map_dbl(1:p, function(i)
         if (sign(direction[i]) > 0)
            max(pts[, i]) + 5
         else
            min(pts[, i]) - 5) # add a number so
   ideal <- purrr::map_dbl(1:p, function(i) if (sign(direction[i]) < 0) max(pts[, i]) else min(pts[, i]))
   ## project on box so pts are the first rows and rest projections including upper corner points
   set <- as.matrix(pts)
   n <- nrow(set)
   set <- rep(1, p + 1) %x% set  # repeat p + 1 times
   for (i in 1:p) {
      set[(i * n + 1):((i + 1) * n), i] <- nadir[i]
   }
   # find upper corner points of box
   cP <- matrix(rep(nadir, p), byrow = T, ncol = p)   # repeat p + 1 times
   diag(cP) <- ideal
   cP <- rbind(cP, nadir)
   # merge and tidy
   set <- rbind(set, cP)
   set <- set[!duplicated(set, MARGIN = 1), ]
   # find hull of the unique points and classify
   set <- convexHull(set, addRays = FALSE, direction = direction)
   # hull <- set$hull
   set <- set$pts
   set$pt[(n+1):length(set$pt)] <- 0
   colnames(set)[1:p] <- paste0("z", 1:p)
   set <- set %>% # tidy and add old id
      dplyr::filter(.data$pt == 1) %>%
      dplyr::mutate(cls = dplyr::if_else(vtx, "se", NA_character_), se = vtx) |> 
      dplyr::select(tidyselect::all_of(1:p), c("cls", "se")) 
   rownames(set) <- NULL
   return(set)
}


#' Update the joint result file with the new results
#'
#' @return Algorithms updated results for.
updateProbStatFile <- function() {
   cat("Update statistics for results.")
   tictoc::tic()
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
   tictoc::toc()
   return(alg)
}


#### Run script ####
## Open log file
# zz <- file(here::here("code/instances/results/calc-stat.log"), open = "wt")
# sink(zz, type = "output", split = T)   # open the file for output
# sink(zz, type = "message")  # open the same file for messages, errors and warnings

paths <- fs::dir_ls(here::here("code/instances/results"), recurse = T, type = "file", glob = "*prob*.json")
datOkay <-  read_csv(here::here("code/instances/stat-prob-okay.csv")) |> # info about what stat already done
   mutate(path = fs::path_file(path)) |> 
   distinct(path, type, alg)
timeLimit <- 120 * 60  # max run time in sec
tictoc::tic.clear()
start <- Sys.time()
cat("\n\nStarting running R script.\n\n")

datCalc <- datOkay |> filter(type == "simple")
idx <- which(fs::path_file(paths) %in% fs::path_file(datCalc$path))
paths1 <- paths[-idx] 
cat("\n\nUpdate simple stat\n\n")
calc <- FALSE
for (path in paths1) {
   calc <- any(calc, calcStat(path))
   cpu <- difftime(Sys.time(), start, units = "secs")
   if (calc) cat("Cpu total", cpu, "\n")
   if (cpu > timeLimit) {
      cat("Time limit reached! Stop R script.")
      break
   }
}
if (calc) updateProbStatFile()
cat("\n\nDone.\n\n")



cat("\n\nClassify extreme points\n\n")
datErrorExt <- read_csv(here::here("code/instances/stat-prob-error.csv")) %>% 
   filter(type == "classifyExt")
idx <- which(fs::path_file(paths) %in% fs::path_file(datErrorExt$path))
paths1 <- paths[-idx]  # paths that we try to classify only extreme in (may already have been)
datCalc <- datOkay |> filter(type == "classifyExt")
idx <- which(fs::path_file(paths1) %in% fs::path_file(datCalc$path))
if (length(idx) > 0) paths1 <- paths1[-idx] 
calc <- FALSE
if (cpu < timeLimit) {
   for (path in fs::path_file(paths1)) {
      res <- tryCatchLog(classifyStat(here::here(str_c("code/instances/results/algorithm1/", path)), classifyExt = T), 
                         error = function(c) {
                            datError <- bind_rows(datError, c(path = path, type = "classifyExt", alg = "alg1"))
                            write_csv(datError, file = here::here("code/instances/stat-prob-error.csv"))
                            return(NA)
                         })
      if (is.na(res)) break   # stop so can commit
      calc <- any(calc, res)
      cpu <- difftime(Sys.time(), start, units = "secs")
      cat("Cpu total", cpu, "\n")
      if (cpu > timeLimit) {
         cat("Time limit reached! Stop R script.")
         break
      }
   }
   if (calc) updateProbStatFile()
}
cat("\n\nDone.\n\n")



cat("\n\nClassify fully\n\n")
datErrorCls <- read_csv(here::here("code/instances/stat-prob-error.csv")) %>% 
   filter(type %in% c("classify", "classifyExt"))
idx <- which(fs::path_file(paths) %in% fs::path_file(datErrorCls$path))
paths1 <- paths[-idx]  # paths that we try to classify (may already have been)
datCalc <- datOkay |> filter(type == "classify")
idx <- which(fs::path_file(paths1) %in% fs::path_file(datCalc$path))
if (length(idx) > 0) paths1 <- paths1[-idx] 
calc <- FALSE
if (cpu < timeLimit) {
   for (path in paths1) {
      res <- tryCatchLog(classifyStat(path), 
         error = function(c) {
            datError <- bind_rows(datError, c(path = path, type = "classify", alg = "alg1"))
            write_csv(datError, file = here::here("code/instances/stat-prob-error.csv"))
            return(NA)
         })
      if (is.na(res)) break   # stop so can commit
      calc <- any(calc, res)
      cpu <- difftime(Sys.time(), start, units = "secs")
      cat("Cpu total", cpu, "\n")
      if (cpu > timeLimit) {
         cat("Time limit reached! Stop R script.")
         break
      }
   }
   if (calc) updateProbStatFile()
}
cat("\n\nDone.\n\n")

cat("\n\nFinish running R script.\n\n")

## Close log file
# sink(type = "message")  # close the file for output
# sink()  # close the file for messages, errors and warnings


#### Tests ####
# path <- paths[1]
# calcStat(path)
# 
# pts[duplicated(pts),]
# pts$nd <- T
# plotly::plot_ly(pts, x = ~z1, y = ~z2)


# 
# for (path in paths) {
#    lst <- jsonlite::read_json(path, simplifyVector = T)
#    p <- lst$statistics$p
#    pts <- classifyNDSet(lst$points[, 1:p]);
#    cat("se:", sum(pts$se),"\n")
#    pts <- classify(lst$points[, 1:p]);
#    cat("se:", sum(pts$se),"\n")
# }
