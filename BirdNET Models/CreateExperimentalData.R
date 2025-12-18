################################################################################
# Reverse Grid Builder (clean version using D as truth for CallType + Category)
################################################################################

rm(list = ls())
set.seed(5)

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(purrr)
  library(ggplot2)
  library(tidyr)
  library(forcats)
  library(scales)
})

# -----------------------------
# Config & Paths
# -----------------------------
in_csv  <- "C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\BirdNetGrid2/DCLDE_train_parent_birdnetGrid2.csv"
out_dir <- "C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdnetOrganized\\BirdNetGrid2"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

N_PER_CLASS <- 3000L
KW_LABELS   <- c("SRKW", "TKW")

# -----------------------------
# Load & Preprocess
# -----------------------------
D <- read_csv(in_csv, show_col_types = FALSE)

D <- D %>%
  filter(
    Provider != "UAF",                      # redundant but safe
    KW_certain %in% c(1, NA),
    Labels %in% c("Background", "HW", "SRKW", "TKW")
  ) %>%
  mutate(
    # DO NOT touch CalltypeCategory mapping here: it is already correct from upstream
    ID        = row_number(),
    Duration  = ifelse(!is.na(FileEndSec) & !is.na(FileBeginSec),
                       FileEndSec - FileBeginSec, NA_real_),
    HasQ      = ifelse(is.na(CallType), FALSE, str_detect(CallType, "\\?")),
    # Only turn NA CallType into "" so we don't blow up on string ops
    CallType  = ifelse(is.na(CallType), "", CallType),
    Augmented = FALSE,
    ShiftSec  = 0,
    SourceID  = ID
  )

# -------------------------------------------------------------------
# Derive repertoire tokens FROM DATA for SRKW/TKW (no hard-coded T13)
# -------------------------------------------------------------------
srkw_tokens <- D %>%
  filter(Labels == "SRKW",
         !is.na(CalltypeCategory),
         CalltypeCategory != "None") %>%
  pull(CalltypeCategory) %>%
  unique() %>%
  sort()

tkw_tokens <- D %>%
  filter(Labels == "TKW",
         !is.na(CalltypeCategory),
         CalltypeCategory != "None") %>%
  pull(CalltypeCategory) %>%
  unique() %>%
  sort()

kw_cats <- list(
  SRKW = srkw_tokens,
  TKW  = tkw_tokens
)

cat("SRKW tokens:\n"); print(srkw_tokens)
cat("TKW tokens:\n");  print(tkw_tokens)


# -------------------------------------------------------------------
# Typed / untyped pools using existing CalltypeCategory in D
# -------------------------------------------------------------------
# Typed pool = rows with a non-None calltypeCategory and no '?'
D_kw_typed <- D %>%
  filter(
    Labels %in% KW_LABELS,
    !is.na(CalltypeCategory),
    CalltypeCategory != "None",
    !HasQ
  )

TYPED_SRKW <- D_kw_typed %>% filter(Labels == "SRKW")
TYPED_TKW  <- D_kw_typed %>% filter(Labels == "TKW")

# Untyped pools = ecotype rows with Category == "None" or NA
UNTYPED_SRKW <- D %>%
  filter(Labels == "SRKW",
         is.na(CalltypeCategory) | CalltypeCategory == "None")

UNTYPED_TKW  <- D %>%
  filter(Labels == "TKW",
         is.na(CalltypeCategory) | CalltypeCategory == "None")

# -------------------------------------------------------------------
# Provider-stratified sampler (unchanged logic)
# -------------------------------------------------------------------
stratified_sample_n <- function(df, group_col = "Provider", n_total) {
  if (nrow(df) == 0 || n_total <= 0) return(df[0, ])
  
  df2 <- df %>%
    mutate(.grp = ifelse(is.na(.data[[group_col]]),
                         "__NA__",
                         as.character(.data[[group_col]])))
  
  gtab <- df2 %>% count(.grp, name = "n")
  
  if (sum(gtab$n) < n_total) {
    return(df2 %>%
             slice_sample(n = n_total, replace = TRUE) %>%
             select(-.grp))
  }
  
  props <- gtab %>%
    mutate(target = pmax(0L, round(n_total * n / sum(n))))
  
  drift <- n_total - sum(props$target)
  if (drift != 0) {
    ord <- order(-props$n)
    ix  <- rep(ord, length.out = abs(drift))
    props$target[ix] <- props$target[ix] + sign(drift)
  }
  
  sampled <- df2 %>%
    left_join(props, by = ".grp") %>%
    group_by(.grp) %>%
    group_modify(~ {
      k <- .x$target[1]
      if (is.na(k) || k <= 0) return(.x[0, ])
      if (nrow(.x) <= k) .x else dplyr::slice_sample(.x, n = k)
    }) %>%
    ungroup()
  
  have <- nrow(sampled)
  if (have == n_total) {
    return(sampled %>% select(-any_of(c(".grp","n","target"))))
  }
  
  need      <- n_total - have
  remaining <- df2 %>% anti_join(sampled %>% select(ID), by = "ID")
  cap       <- remaining %>% count(.grp, name = "cap")
  cap <- cap %>%
    mutate(take = pmax(0L, floor(need * cap / sum(cap))))
  
  drift2 <- need - sum(cap$take)
  if (drift2 != 0) {
    ord2 <- order(-cap$cap)
    ix2  <- rep(ord2, length.out = abs(drift2))
    cap$take[ix2] <- cap$take[ix2] + sign(drift2)
  }
  
  topup <- remaining %>%
    inner_join(cap %>% select(.grp, take), by = ".grp") %>%
    group_by(.grp) %>%
    group_modify(~ {
      k <- .y$take[1]
      if (k <= 0) return(.x[0, ])
      if (nrow(.x) <= k) .x else dplyr::slice_sample(.x, n = k)
    }) %>%
    ungroup()
  
  sampled <- bind_rows(sampled, topup)
  if (nrow(sampled) > n_total) {
    sampled <- dplyr::slice_sample(sampled, n = n_total)
  }
  
  sampled %>%
    select(-any_of(c(".grp","n","target","cap","take")))
}

# -------------------------------------------------------------------
# Fixed BG & HW (identical across 9) — KW-free background
# -------------------------------------------------------------------
kw_tokens_all   <- c(kw_cats$SRKW, kw_cats$TKW)
kw_exact_regex  <- paste0("(", paste(stringr::str_c("\\b", kw_tokens_all, "\\b"),
                                     collapse = "|"), ")")
kw_generic_regex <- "(\\bS\\d{2}\\b|\\bT\\d{2}\\b)"
kw_any_regex     <- paste0("(", kw_exact_regex, "|", kw_generic_regex, ")")

BG_pool <- D %>%
  filter(
    Labels == "Background",
    ClassSpecies %in% c("AB", "UndBio"),
    !(Ecotype %in% c("SRKW","TKW"))
  ) %>%
  filter(is.na(CallType) | CallType == "" | !stringr::str_detect(CallType, kw_any_regex))

stopifnot(nrow(BG_pool) >= N_PER_CLASS)

# Here we intentionally set CalltypeCategory = "None" for BG/HW
BG_FIXED <- stratified_sample_n(BG_pool, "Provider", N_PER_CLASS) %>%
  mutate(Labels = "Background",
         CalltypeCategory = "None")

HW_pool <- D %>% filter(Labels == "HW")
stopifnot(nrow(HW_pool) >= N_PER_CLASS)

HW_FIXED <- stratified_sample_n(HW_pool, "Provider", N_PER_CLASS) %>%
  mutate(Labels = "HW",
         CalltypeCategory = "None")

# Sanity: no KW-like tokens in BG/HW CallType
kw_in_bg <- BG_FIXED %>%
  filter(!is.na(CallType) & CallType != "" &
           stringr::str_detect(CallType, kw_any_regex))
kw_in_hw <- HW_FIXED %>%
  filter(!is.na(CallType) & CallType != "" &
           stringr::str_detect(CallType, kw_any_regex))
stopifnot(nrow(kw_in_bg) == 0, nrow(kw_in_hw) == 0)

# -------------------------------------------------------------------
# Augmentation (does not modify CallType, only times + ID + flags)
# -------------------------------------------------------------------
.AUG_ID_BASE <- max(D$ID, na.rm = TRUE)
.AUG_ID_INC  <- 0L

augment_rows <- function(df, need) {
  if (need <= 0) return(df[0, ])
  
  base <- df %>%
    slice_sample(n = need, replace = TRUE)
  
  shifts <- floor(runif(need, -50, 50)) / 100 * base$Duration
  
  base$FileBeginSec <- base$FileBeginSec + shifts
  base$FileEndSec   <- base$FileEndSec   + shifts
  base$Augmented    <- TRUE
  base$ShiftSec     <- shifts
  base$SourceID     <- base$ID
  base$ID           <- .AUG_ID_BASE + seq_len(need) + .AUG_ID_INC
  
  .AUG_ID_INC <<- .AUG_ID_INC + need
  base
}

# -------------------------------------------------------------------
# Make base (untyped-only) ecotype sets
# -------------------------------------------------------------------
make_base_untyped <- function(ecotype, untyped_pool) {
  stratified_sample_n(untyped_pool, "Provider", N_PER_CLASS) %>%
    mutate(
      Labels = ecotype,
      # These *should* already be "None", but enforce for clarity
      CalltypeCategory = "None"
    )
}

BASE_SRKW <- make_base_untyped("SRKW", UNTYPED_SRKW)
BASE_TKW  <- make_base_untyped("TKW",  UNTYPED_TKW)

# -------------------------------------------------------------------
# Equal-per-CT replacer for a target typed total (e.g., 300 or 900)
# -------------------------------------------------------------------
equal_split_targets <- function(total_needed, tokens) {
  K <- length(tokens)
  if (K == 0) stop("No repertoire tokens.")
  base_q <- floor(total_needed / K)
  rem_q  <- total_needed - base_q * K
  
  tgt <- rep(base_q, K)
  if (rem_q > 0) {
    tgt[seq_len(rem_q)] <- tgt[seq_len(rem_q)] + 1L
  }
  
  tibble::tibble(CalltypeCategory = tokens,
                 target_n        = tgt)
}

per_ct_draw <- function(ecotype, ct, need, typed_pool_all) {
  if (need <= 0) return(typed_pool_all[0, ])
  
  avail     <- typed_pool_all %>% filter(CalltypeCategory == ct)
  take_real <- min(nrow(avail), need)
  
  real_rows <- if (take_real > 0) {
    stratified_sample_n(avail, "Provider", take_real)
  } else {
    avail[0, ]
  }
  
  short <- need - take_real
  
  if (short > 0) {
    if (nrow(avail) > 0) {
      aug <- augment_rows(avail, short)
      aug$AugFromCT <- ct
      # CalltypeCategory already == ct
    } else {
      # ZERO seed: borrow from the largest CT for this ecotype.
      # Note: ct is guaranteed to be in kw_cats[[ecotype]] which
      # is derived from D, so you won't get phantom categories.
      cap   <- typed_pool_all %>% count(CalltypeCategory, name = "cap")
      donor <- cap$CalltypeCategory[which.max(cap$cap)]
      donor_pool <- typed_pool_all %>% filter(CalltypeCategory == donor)
      aug <- augment_rows(donor_pool, short)
      aug$AugFromCT      <- donor
      aug$CalltypeCategory <- ct   # re-label donor examples as target CT
      message(sprintf("Warning: no real '%s' for %s; augmented %d from donor '%s'.",
                      ct, ecotype, short, donor))
    }
    bind_rows(real_rows, aug)
  } else {
    real_rows$AugFromCT <- NA_character_
    real_rows
  }
}

replace_with_equal_ct <- function(prev_df, ecotype, typed_pool_all, pct) {
  target_total   <- round(pct * N_PER_CLASS)
  is_typed       <- prev_df$CalltypeCategory != "None" & !is.na(prev_df$CalltypeCategory)
  current_typed  <- sum(is_typed)
  if (current_typed == target_total) return(prev_df)
  
  tokens <- kw_cats[[ecotype]]   # from data, not hard-coded
  
  curr_ct <- prev_df %>%
    filter(Labels == ecotype,
           CalltypeCategory != "None") %>%
    count(CalltypeCategory, name = "n") %>%
    right_join(tibble::tibble(CalltypeCategory = tokens),
               by = "CalltypeCategory") %>%
    mutate(n = replace_na(n, 0L)) %>%
    arrange(CalltypeCategory)
  
  targets <- equal_split_targets(target_total, tokens)
  
  need_tbl <- curr_ct %>%
    left_join(targets, by = "CalltypeCategory") %>%
    mutate(need = pmax(0L, target_n - n))
  
  add_list <- purrr::map2(
    need_tbl$CalltypeCategory,
    need_tbl$need,
    ~ per_ct_draw(ecotype, .x, .y, typed_pool_all)
  )
  
  add_typed <- bind_rows(add_list)
  
  replace_n  <- nrow(add_typed)
  untyped_ids <- prev_df$ID[prev_df$CalltypeCategory == "None" |
                              is.na(prev_df$CalltypeCategory)]
  
  if (length(untyped_ids) < replace_n) {
    extra     <- replace_n - length(untyped_ids)
    typed_ids <- prev_df$ID[prev_df$CalltypeCategory != "None" &
                              !is.na(prev_df$CalltypeCategory)]
    replace_ids <- c(untyped_ids, sample(typed_ids, extra))
  } else {
    replace_ids <- sample(untyped_ids, replace_n)
  }
  
  keep_df <- prev_df %>%
    filter(!(ID %in% replace_ids))
  
  out <- bind_rows(keep_df, add_typed) %>%
    mutate(CalltypeCategory = ifelse(is.na(CalltypeCategory) |
                                       CalltypeCategory == "",
                                     "None",
                                     CalltypeCategory)) %>%
    arrange(ID)
  
  out
}

# -------------------------------------------------------------------
# Build SRKW/TKW levels: 0%, then 10% from 0, then 30% from 10
# -------------------------------------------------------------------
SRKW_0  <- BASE_SRKW
SRKW_10 <- replace_with_equal_ct(SRKW_0,  "SRKW", TYPED_SRKW, 0.10)
SRKW_30 <- replace_with_equal_ct(SRKW_10, "SRKW", TYPED_SRKW, 0.30)

TKW_0  <- BASE_TKW
TKW_10 <- replace_with_equal_ct(TKW_0,  "TKW",  TYPED_TKW,  0.10)
TKW_30 <- replace_with_equal_ct(TKW_10, "TKW",  TYPED_TKW,  0.30)

# -------------------------------------------------------------------
# Console diagnostics
# -------------------------------------------------------------------
plot_stacked_by_label_counts <- function(df, title_suffix) {
  df2 <- df %>%
    mutate(CalltypeCategory = ifelse(is.na(CalltypeCategory) |
                                       CalltypeCategory == "",
                                     "None",
                                     CalltypeCategory))
  lvls <- c(setdiff(sort(unique(df2$CalltypeCategory)), "None"), "None")
  df2$CalltypeCategory <- factor(df2$CalltypeCategory, levels = lvls)
  
  df2 %>%
    count(Labels, CalltypeCategory, name = "n") %>%
    ggplot(aes(x = Labels, y = n, fill = CalltypeCategory)) +
    geom_col(width = 0.75, color = "gray15", linewidth = 0.2) +
    labs(title = paste("Class Composition (counts)", title_suffix),
         x = NULL, y = "Count", fill = "Call type") +
    theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank(),
          axis.text.x = element_text(size = 11)) %>%
    print()
}

plot_stacked_by_label_props <- function(df, title_suffix) {
  df2 <- df %>%
    mutate(CalltypeCategory = ifelse(is.na(CalltypeCategory) |
                                       CalltypeCategory == "",
                                     "None",
                                     CalltypeCategory))
  lvls <- c(setdiff(sort(unique(df2$CalltypeCategory)), "None"), "None")
  df2$CalltypeCategory <- factor(df2$CalltypeCategory, levels = lvls)
  
  df2 %>%
    count(Labels, CalltypeCategory, name = "n") %>%
    group_by(Labels) %>%
    mutate(prop = n / sum(n)) %>%
    ungroup() %>%
    ggplot(aes(x = Labels, y = prop, fill = CalltypeCategory)) +
    geom_col(width = 0.75, color = "gray15", linewidth = 0.2) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(title = paste("Class Composition (proportions)", title_suffix),
         x = NULL, y = "Proportion", fill = "Call type") +
    theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank(),
          axis.text.x = element_text(size = 11)) %>%
    print()
}

# -------------------------------------------------------------------
# Assemble & write all 9 experiments (+ plots)
# -------------------------------------------------------------------
grid <- list(
  "SRKW0_TKW0"   = list(SRKW_0,  TKW_0),
  "SRKW0_TKW10"  = list(SRKW_0,  TKW_10),
  "SRKW0_TKW30"  = list(SRKW_0,  TKW_30),
  "SRKW10_TKW0"  = list(SRKW_10, TKW_0),
  "SRKW10_TKW10" = list(SRKW_10, TKW_10),
  "SRKW10_TKW30" = list(SRKW_10, TKW_30),
  "SRKW30_TKW0"  = list(SRKW_30, TKW_0),
  "SRKW30_TKW10" = list(SRKW_30, TKW_10),
  "SRKW30_TKW30" = list(SRKW_30, TKW_30)
)

for (nm in names(grid)) {
  SR <- grid[[nm]][[1]]
  TK <- grid[[nm]][[2]]
  
  final_df <- bind_rows(BG_FIXED, HW_FIXED, SR, TK) %>%
    mutate(CalltypeCategory = ifelse(is.na(CalltypeCategory) |
                                       CalltypeCategory == "",
                                     "None",
                                     CalltypeCategory))
  
  # All 4 classes should have exactly N_PER_CLASS examples
  stopifnot(all(table(final_df$Labels) == N_PER_CLASS))
  
  
  out_file <- file.path(out_dir, paste0("Grid_", nm, ".csv"))
  write_csv(final_df, out_file)
  message("Wrote ", out_file)
  
  plot_stacked_by_label_counts(final_df, nm)
  plot_stacked_by_label_props(final_df,  nm)
}
