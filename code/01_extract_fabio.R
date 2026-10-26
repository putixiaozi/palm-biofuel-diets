# Extract vegetable-oil, oilseed and oilcake flows from FABIO v2 beta (2010-2023)
suppressMessages(library(Matrix))
fab <- Sys.getenv("FABIO_DIR", "data_fabio/")
out <- "data_derived/"
dir.create(out, showWarnings = FALSE)

io  <- read.csv(paste0(fab, "io_labels.csv"))
su  <- read.csv(paste0(fab, "su_labels.csv"))
fdl <- read.csv(paste0(fab, "fd_labels.csv"))
oils  <- sprintf("c%03d", 68:80)
seeds <- c(sprintf("c%03d", 21:30), "c063", "c064")
cakes <- sprintf("c%03d", 81:89)
crushp <- sprintf("p%03d", 67:79)

Y <- readRDS(paste0(fab, "Y.rds"))
Z <- readRDS(paste0(fab, "Z_mass.rds"))
X <- readRDS(paste0(fab, "X.rds"))
S <- readRDS(paste0(fab, "mr_sup_mass.rds"))
U <- readRDS(paste0(fab, "mr_use.rds"))

ri <- which(io$comm_code %in% oils)
bil <- list(); bal <- list(); cop <- list()
for (yr in names(Y)) {
  y <- Y[[yr]][ri, , drop = FALSE]
  t <- summary(as(y, "TsparseMatrix"))
  b1 <- data.frame(year = as.integer(yr), from = io$iso3c[ri][t$i], item = io$comm_code[ri][t$i],
                   to = fdl$iso3c[t$j], use = fdl$fd[t$j], value = t$x)
  z <- Z[[yr]][ri, , drop = FALSE]
  t <- summary(as(z, "TsparseMatrix"))
  b2 <- data.frame(year = as.integer(yr), from = io$iso3c[ri][t$i], item = io$comm_code[ri][t$i],
                   to = io$iso3c[t$j], use = paste0("int_", io$comm_code[t$j]), value = t$x)
  b2 <- aggregate(value ~ year + from + item + to + use, b2, sum)
  bil[[yr]] <- rbind(b1, b2)
  # production (total output) of oils, seeds, cakes
  xi <- which(io$comm_code %in% c(oils, seeds, cakes))
  x <- X[, yr]
  bal[[yr]] <- data.frame(year = as.integer(yr), iso3c = io$iso3c[xi], item = io$comm_code[xi], X = as.numeric(x[xi]))
  # co-production: supply of crushing processes and their seed inputs
  pj <- which(su$proc_code %in% crushp)
  s <- S[[yr]][pj, , drop = FALSE]; t <- summary(as(s, "TsparseMatrix"))
  c1 <- data.frame(year = as.integer(yr), iso3c = su$iso3c[pj][t$i], proc = su$proc_code[pj][t$i],
                   item = io$comm_code[t$j], flow = "output", value = t$x)
  u <- U[[yr]][, pj, drop = FALSE]; t <- summary(as(u, "TsparseMatrix"))
  c2 <- data.frame(year = as.integer(yr), iso3c = su$iso3c[pj][t$j], proc = su$proc_code[pj][t$j],
                   item = io$comm_code[t$i], flow = "input", value = t$x)
  c2 <- aggregate(value ~ year + iso3c + proc + item + flow, c2, sum)
  cop[[yr]] <- rbind(c1, c2)
  cat(yr, nrow(bil[[yr]]), "\n")
}
write.csv(do.call(rbind, bil), gzfile(paste0(out, "fabio_oil_bilateral_use_2010_2023.csv.gz")), row.names = FALSE)
write.csv(do.call(rbind, bal), paste0(out, "fabio_oil_seed_cake_output_2010_2023.csv"), row.names = FALSE)
write.csv(do.call(rbind, cop), paste0(out, "fabio_crushing_coproduction_2010_2023.csv"), row.names = FALSE)
