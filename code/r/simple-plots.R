library(tidyverse)
library(gMOIP)

#### Functions ####

#' Sum points from the subproblems
#' @param idx Index of vectors to sum.
#' @return The sum y.
addP <- function(idx) {  
   v <- subP[[1]][idx[1],]
   for (s in 2:length(idx)) {
      v <- v + subP[[s]][idx[s],]
   }
   return(v)
}

#' Find the Minkowski sum
#'
#' @param subP List of subproblems.
#' @return The MS with classification and combinations.
msY <- function(subP) {
   Y <- expand.grid(map(subP, function(x) 1:nrow(x))) %>% 
      rowwise() %>% 
      mutate(idx = list(c_across(everything())), y = list(addP(idx))) %>% 
      select(-contains("Var")) %>% 
      unnest_wider(y, names_sep = "") %>%
      group_by(across(contains("y"))) %>% 
      nest() %>% 
      mutate(text = map_chr(data, function(df) {
         df %>% 
            rowwise() %>% 
            mutate(str = str_c(idx, collapse=',')) %>% 
            mutate(str = str_c("(", str, ")")) %>% 
            pull(str) %>% str_c(collapse = ", ") 
      })) %>% 
      mutate(lgd = map_dbl(data, function(df) {
         nrow(df)
      })) %>% 
      ungroup()#%>% select(data) #%>% view() 
   dat <- addNDSet(Y[,1:p], keepDom = T, crit = "min")  # classify points
   Y$cls <- dat$cls
   return(Y)
}


#' Plot the Minkowski sum
#'
#' @param Y The MS.
#' @param subP List with subproblems.
#' @return A ggplot.
plotMS <- function(Y, subP, plotDom = TRUE) {
   # First add Y
   dat <- addNDSet(Y[,1:p], keepDom = T, crit = "min") %>% mutate(prob = "ms")
   dat$text <- Y$text
   # Next add subP
   m <- length(subP)
   for (s in 1:m) {
      dat <- bind_rows(dat,
                       addNDSet(subP[[s]], crit = "min") %>% mutate(text = as.character(1:n()), prob = as.character(s)))
   }
   if (plotDom) datD <- dat %>% filter(cls == "d")
   dat <- dat %>% filter(cls != "d")
   pt <- dat %>% ggplot(aes(x = z1, y = z2, label = text, shape = cls, color = prob)) +
      ggrepel::geom_text_repel(size = 3, 
                               hjust = "left",
                               nudge_x = 0.2, nudge_y = 0.4, 
                               segment.colour=NA, data = dat %>% filter(prob == "ms")) +
      ggrepel::geom_text_repel(size = 2, segment.colour=NA, data = dat %>% filter(prob != "ms")) +
      geom_point() +
      theme_bw()
   if (plotDom) pt <- pt +
      geom_point(data = datD, color = "grey80") +
      ggrepel::geom_text_repel(size = 3, segment.colour=NA, data = datD, color = "grey80", nudge_x = 0.2, nudge_y = -0.1, hjust = "left")
   return(pt)
}

#### Example for paper ####

p <- 2
# Y1 <- matrix(  # n x p matrix
#    c(0, 14,
#      2, 6,
#      4, 5,
#      5, 3,
#      6, 2,
#      12, 0
#      ), ncol = p, byrow = T)
# Y2 <- matrix(  # n x p matrix
#    c(2, 12,
#      4, 8,
#      6, 6,
#      8, 4,
#      15, 0
#    ), ncol = p, byrow = T)
Y1 <- matrix(  # n x p matrix
   c(0, 7,
      2, 6,
     4, 5,
     6, 4,
     9, 3,
    10, 2,
    20, 0
   ), ncol = p, byrow = T)
Y2 <- matrix(  # n x p matrix
   c(12, 4,
     14, 3,
     16, 2
   ), ncol = p, byrow = T)
# Y1 <- matrix(  # n x p matrix
#    c(2, 6,
#      4, 5,
#      6, 4,
#      10, 2
#    ), ncol = p, byrow = T)
# Y2 <- matrix(  # n x p matrix
#    c(12, 4,
#      14, 3,
#      16, 2
#    ), ncol = p, byrow = T)
# Y1 <- matrix(  # n x p matrix
#    c(0, 2,
#      1, 1,
#      2, 0
#    ), ncol = p, byrow = T)
# Y2 <- matrix(  # n x p matrix
#    c(0, 4,
#      1, 3,
#      2, 2,
#      3, 1,
#      4, 0
#    ), ncol = p, byrow = T)
subP <- list(Y1, Y2)
Y <- msY(subP)
plotMS(Y, subP, plotDom = T)




#### General example ####

p <- 2
# Y1 <- matrix(  # n x p matrix
#    c(0, 10,
#      2, 4,
#      4, 2,
#      15, 0
#      ), ncol = p, byrow = T) 
# Y1 <- matrix(  # n x p matrix
#    c(1, 47,
#      2, 41,
#      3, 36,
#      7, 27,
#      10, 22
#    ), ncol = p, byrow = T) 
# Y1 <- matrix(  # n x p matrix
#    c(1, 0,
#      0, 1
#    ), ncol = p, byrow = T)
# Y2 <- Y1
Y1 <- genNDSet(p, 10, range = c(50, 100)) 
# Y2 <- Y1 %>% mutate(z2 = 0.2 * z2)
# Y1 <- Y1 %>% mutate(z1 = 0.2 * z1)
Y1 <- Y1 %>% arrange(across(starts_with("z")))
Y1 <- as.matrix(Y1[,1:p])
## Y2 sphere
Y2 <- genNDSet(p, 5, range = c(0, 50)) 
Y2 <- Y2 %>% arrange(across(starts_with("z")))
Y2 <- as.matrix(Y2[,1:p])
# cent <- c(30,30)
# r <- 20
# planeC <- c(cent+r/3)
# planeC <- c(planeC, -sum(planeC^2))
# Y2 <- genNDSet(2, 10,
#                  argsSphere = list(center = cent, radius = r, below = T, plane = planeC, factor = 6))
## Y2 random
# Y2 <- genNDSet(2, 10, random = T)
# Y2 <- Y2 %>% arrange(across(starts_with("z")))
# Y2 <- as.matrix(Y2[,1:p])
## Y2 switch
# Y2 <- Y1
# Y2[, 1] <- Y1[, 2]
# Y2[, 2] <- Y1[, 1]
# Y2 <- genNDSet(p, 20, planes = T, range = c(0,300)) 
# Y2 <- Y2 %>% arrange(across(starts_with("z")))
# Y2 <- as.matrix(Y2[,1:p])
# Y2 <- Y1
# Y3 <- Y1
Y3 <- genNDSet(p, 5) 
Y3 <- Y3 %>% arrange(across(starts_with("z")))
Y3 <- as.matrix(Y3[,1:p])
Y4 <- genNDSet(p, 5) 
Y4 <- Y4 %>% arrange(across(starts_with("z")))
Y4 <- as.matrix(Y4[,1:p])

subP <- list(Y1, Y2, Y3, Y4)
m <- length(subP)



Y <- expand.grid(map(subP, function(x) 1:nrow(x))) %>% 
   rowwise() %>% 
   mutate(idx = list(c_across(everything())), y = list(addP(idx))) %>% 
   select(-contains("Var")) %>% 
   unnest_wider(y, names_sep = "") %>%
   group_by(across(contains("y"))) %>% 
   nest() %>% 
   mutate(text = map_chr(data, function(df) {
      df %>% 
         rowwise() %>% 
         mutate(str = str_c(idx, collapse=',')) %>% 
         mutate(str = str_c("(", str, ")")) %>% 
         pull(str) %>% str_c(collapse = ", ") 
   })) %>% 
   mutate(lgd = map_dbl(data, function(df) {
      nrow(df)
   })) %>% 
   ungroup()#%>% select(data) #%>% view() 
dat <- addNDSet(Y[,1:p], keepDom = T, crit = "min") %>% mutate(prob = "m")
Y$cls <- dat$cls
view(Y)

## check if all points used in a singleton
singletons <- Y %>% filter(lgd == 1, cls == "se") %>% unnest(data) %>% unnest_wider(idx, names_sep = "") %>% 
   ungroup() %>% 
   select(starts_with("idx")) %>% as.list() %>% map(unique) %>% map(sort)

for (s in 1:m) {
   dat <- addNDSet(subP[[s]], crit = "min")
   idx <- setdiff(1:nrow(subP[[s]]), singletons[[s]])
   if (length(idx) > 0) {
      for (i in idx) {
         cat("Point", subP[[s]][i,], dat$cls[i], "with index", i, "in subp", s, "is not used in an extreme singleton\n") 
      }
   }
}

dat <- addNDSet(Y[,1:p], keepDom = T, crit = "min") %>% mutate(prob = "m")
dat$text <- Y$text

# ini3D()
# plotHull3D(pts, drawPoints = TRUE, addRays = TRUE)
# finalize3D()





for (s in 1:m) {
   dat <- bind_rows(dat, 
      addNDSet(subP[[s]], crit = "min") %>% 
         mutate(text = as.character(1:n()), prob = as.character(s)))
}
dat <- dat %>% filter(cls != "d")
dat %>% ggplot(aes(x = z1, y = z2, label = text, shape = cls, color = prob)) +
   ggrepel::geom_text_repel(size = 3, colour = "gray50", nudge_x = 4, nudge_y = 4, segment.colour="gray90", data = dat %>% filter(prob == "m")) +
   ggrepel::geom_text_repel(size = 2, colour = "gray50", segment.colour="gray90", data = dat %>% filter(prob != "m")) +
   geom_point() 

