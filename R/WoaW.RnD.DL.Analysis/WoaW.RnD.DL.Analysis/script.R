dispaly.result <- function(fileName) {
    data <- read.table(fileName, FALSE, sep = " ");

    e1 <- data[[4]] - data[[3]];
    w1 <- data[[1]];
    e2 <- (data[[4]] - data[[3]]) ^ 2 / length(data);

    old.par <- par(mfrow = c(1, 2))
    plot(w1, e2, type = "p", col = "green")
    plot(w1, e1, type = "p", col = "red")
    par(old.par)


    #https://stackoverflow.com/questions/2564258/plot-two-graphs-in-same-plot-in-r
}
