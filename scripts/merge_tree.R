## ============================================================
## plot_merge_tree.R
## Dendrogram of the greedy cluster merging from run_merge.py.
##
## Leaves are the K_init starting clusters. Going up, each join is one merge step;
## the height of a join is the cumulative loss of ELBO (model fit) caused by the merges
## so far, so clusters joined low down are similar and joins high up cost a lot of fit.
## Branches are coloured by cell type when every cluster below them shares that type.
##
## Inspect merge output to find the resolution at which DBI is lowest for stable fit.
##
## Run after:  python scripts/run_merge.py --data-dir ... --out-dir ... --seed 1
## ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(stringr)
})

## -----------------------------
## Configuration
## -----------------------------
results_dir <- "results/baron"   # --out-dir used for run_merge.py
seed        <- 1                 # --seed used for run_merge.py (it's in the file names)
K_init      <- 15                # which --k-init run to draw
K_cut       <- 8                 # draw a cut line where K_cut clusters remain (NULL for none)
out_path    <- file.path(results_dir, sprintf("merge_tree_Kinit%d_%d.pdf", K_init, seed))

## Naming the leaves. Pick one:
##  (a) cell_type_map = NULL and a Seurat object + label column: each starting cluster is
##      named by the most common label among its cells (needs the true_labels/cell_id
##      columns written by run_merge.py)
##  (b) cell_type_map = c("0" = "Naive T cell", "1" = "Classical Mono", ...) by hand
##  (c) cell_type_map = NULL and seurat_path = NULL: leaves are uncoloured
cell_type_map     <- NULL
seurat_path       <- NULL        # e.g. "data/baron/processed.rds"
label_column      <- "celltype"
cell_type_colours <- NULL        # named vector of colours, or NULL for an automatic palette

## -----------------------------
## Load
## -----------------------------
history_df <- read.csv(file.path(results_dir, sprintf("merge_history_%d.csv", seed)))
alloc_df   <- read.csv(file.path(results_dir, sprintf("cluster_allocations_%d.csv", seed)),
                       check.names = FALSE)
hist_k <- history_df %>% filter(K_init == !!K_init) %>% arrange(desc(K))
if (nrow(hist_k) == 0) stop("No merge history for K_init = ", K_init)

seu <- if (!is.null(seurat_path)) readRDS(seurat_path) else NULL

## -----------------------------
## Name each starting cluster by its majority label
## -----------------------------
init_col <- paste0("Kinit", K_init, "_K", max(hist_k$K))
if (is.null(cell_type_map) && !is.null(seu) && !is.null(label_column)) {
  labels <- seu@meta.data[[label_column]][match(alloc_df$cell_id, colnames(seu))]
  if (all(is.na(labels))) stop("No cell IDs in cluster_allocations match the Seurat object")
  majority <- data.frame(cluster = alloc_df[[init_col]], label = as.character(labels)) %>%
    filter(!is.na(label)) %>%
    count(cluster, label) %>%
    group_by(cluster) %>%
    mutate(purity = n / sum(n)) %>%
    slice_max(n, n = 1, with_ties = FALSE) %>%
    ungroup()
  message("Majority label per starting cluster (purity = fraction of cells with that label):")
  print(as.data.frame(majority))
  cell_type_map <- setNames(majority$label, as.character(majority$cluster))
}

## -----------------------------
## Rebuild the merge tree
## -----------------------------
## merged_pair "(j, k)" gives positions in the list of clusters still active at that step;
## cluster k is folded into cluster j and removed, so later positions shift down by one.
build_merge_tree <- function(hist_k) {
  K_start <- max(hist_k$K)
  merges  <- hist_k %>% filter(!is.na(merged_pair), merged_pair != "")
  if (nrow(merges) != nrow(hist_k) - 1) {
    stop("Expected one merge per row after the first (", nrow(hist_k) - 1, "), found ", nrow(merges))
  }
  drops   <- pmax(-diff(hist_k$elbo_score), 0)   # ELBO lost at each merge (clipped at 0)
  heights <- cumsum(drops)

  parse_pair <- function(s) as.integer(str_extract_all(s, "[0-9]+")[[1]])
  pairs <- lapply(merges$merged_pair, parse_pair)

  ## If merging stopped at K_final > 1, join the remaining clusters at a nominal height
  ## so the tree has a single root. These joins are drawn dashed: they are not real merges.
  n_real  <- length(pairs)
  n_extra <- min(hist_k$K) - 1
  if (n_extra > 0) {
    step <- max(heights, 1) * 0.05
    for (e in seq_len(n_extra)) {
      pairs[[n_real + e]] <- c(0L, 1L)
      heights <- c(heights, max(heights, 0) + step)
    }
  }

  n_merges  <- length(pairs)
  merge_mat <- matrix(0L, nrow = n_merges, ncol = 2)   # negative = leaf (-(label + 1)), positive = merge
  active    <- 0:(K_start - 1)
  node_of   <- setNames(-(1:K_start), as.character(active))
  for (i in seq_len(n_merges)) {
    pos  <- pairs[[i]] + 1L
    keep <- as.character(active[pos[1]])
    gone <- as.character(active[pos[2]])
    merge_mat[i, ] <- c(node_of[[keep]], node_of[[gone]])
    node_of[[keep]] <- i
    node_of <- node_of[names(node_of) != gone]
    active  <- active[-pos[2]]
  }
  list(merge_mat = merge_mat, heights = heights, is_dummy = seq_len(n_merges) > n_real,
       K_start = K_start)
}

tree <- build_merge_tree(hist_k)
merge_mat <- tree$merge_mat
n_merges  <- nrow(merge_mat)

## Leaf order from a depth-first walk so branches don't cross
leaf_label <- function(node) as.character(-node - 1)
get_leaves <- function(node) {
  if (node < 0) return(leaf_label(node))
  c(get_leaves(merge_mat[node, 1]), get_leaves(merge_mat[node, 2]))
}
leaf_order <- get_leaves(n_merges)
leaf_x     <- setNames(seq_along(leaf_order), leaf_order)

node_x <- numeric(n_merges)
pos_of <- function(node) if (node < 0) leaf_x[[leaf_label(node)]] else node_x[node]
ht_of  <- function(node) if (node < 0) 0 else tree$heights[node]
for (i in seq_len(n_merges)) node_x[i] <- mean(c(pos_of(merge_mat[i, 1]), pos_of(merge_mat[i, 2])))

## A node gets a cell type if everything below it has that type
node_key <- function(node) if (node < 0) leaf_label(node) else paste0("node", node)
type_of  <- list()
for (lab in leaf_order) {
  type_of[[lab]] <- if (!is.null(cell_type_map) && lab %in% names(cell_type_map)) unname(cell_type_map[[lab]]) else NA
}
for (i in seq_len(n_merges)) {
  t1 <- type_of[[node_key(merge_mat[i, 1])]]
  t2 <- type_of[[node_key(merge_mat[i, 2])]]
  type_of[[node_key(i)]] <- if (!is.na(t1) && !is.na(t2) && t1 == t2) t1 else NA
}
node_type <- function(node) type_of[[node_key(node)]]

## Segments: a vertical branch from each child up to its parent, plus the horizontal join
segs <- list()
for (i in seq_len(n_merges)) {
  for (child in merge_mat[i, ]) {
    segs[[length(segs) + 1]] <- data.frame(
      x = pos_of(child), xend = pos_of(child), y = ht_of(child), yend = tree$heights[i],
      cell_type = node_type(child), dummy = tree$is_dummy[i]
    )
  }
  segs[[length(segs) + 1]] <- data.frame(
    x = pos_of(merge_mat[i, 1]), xend = pos_of(merge_mat[i, 2]),
    y = tree$heights[i], yend = tree$heights[i],
    cell_type = node_type(i), dummy = tree$is_dummy[i]
  )
}
seg_df <- bind_rows(segs)

leaf_df <- data.frame(x = leaf_x, label = names(leaf_x),
                      cell_type = sapply(names(leaf_x), function(l) type_of[[l]]))

## Colours
types <- sort(unique(na.omit(leaf_df$cell_type)))
if (is.null(cell_type_colours) && length(types) > 0) {
  cell_type_colours <- setNames(scales::hue_pal()(length(types)), types)
}

## Cut line halfway between the merge that leaves K_cut clusters and the next one
h_cut <- NULL
if (!is.null(K_cut) && K_cut > min(hist_k$K) && K_cut <= tree$K_start) {
  i_cut <- tree$K_start - K_cut          # number of merges done when K_cut clusters remain
  lower <- if (i_cut == 0) 0 else tree$heights[i_cut]
  h_cut <- (lower + tree$heights[i_cut + 1]) / 2
}

## -----------------------------
## Plot
## -----------------------------
max_h <- max(tree$heights)
p <- ggplot() +
  geom_segment(data = filter(seg_df, is.na(cell_type)),
               aes(x = x, xend = xend, y = y, yend = yend, linetype = dummy),
               colour = "gray30", linewidth = 0.25) +
  geom_segment(data = filter(seg_df, !is.na(cell_type)),
               aes(x = x, xend = xend, y = y, yend = yend, colour = cell_type, linetype = dummy),
               linewidth = 0.6) +
  geom_text(data = leaf_df,
            aes(x = x, y = -max_h * 0.015, label = label, colour = cell_type),
            vjust = 1, size = 2.2, show.legend = FALSE) +
  scale_linetype_manual(values = c(`FALSE` = "solid", `TRUE` = "dashed"), guide = "none") +
  scale_y_continuous(name = "Cumulative ELBO drop", expand = expansion(mult = c(0.06, 0.05))) +
  scale_x_continuous(expand = expansion(add = c(0.5, 1.0))) +
  theme_classic(base_size = 8) +
  theme(axis.title.x = element_blank(), axis.text.x = element_blank(),
        axis.ticks.x = element_blank(), axis.line.x = element_blank(),
        legend.key.size = unit(0.3, "cm"))
if (length(types) > 0) {
  p <- p + scale_colour_manual(values = cell_type_colours, name = NULL, na.value = "gray30",
                               guide = guide_legend(override.aes = list(linewidth = 1)))
}
if (!is.null(h_cut)) {
  p <- p +
    geom_hline(yintercept = h_cut, linetype = "dashed", linewidth = 0.2, colour = "gray60") +
    annotate("text", x = length(leaf_order) + 0.7, y = h_cut, label = paste0("K = ", K_cut),
             size = 2.3, vjust = -0.5, colour = "gray20")
}

ggsave(out_path, p, width = 110, height = 80, units = "mm")
message("Saved: ", out_path)

