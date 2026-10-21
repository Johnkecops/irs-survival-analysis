# ==============================================================================
# PPI NETWORK ANALYSIS: STRING-db & igraph / ggraph
# ==============================================================================
# Author  : Pipeline for Dr. Arli Aditya Parikesit, i3L
#           ORCID: 0000-0001-8716-3926
# Date    : 2026-04-16
#
# Inputs  : DEA_Full_Results_macOS.csv   (DEA – gene symbols + log2FC)
#           GSEA_Results_Detailed.csv    (GSEA – leading-edge ENSG IDs)
#
# Strategy:
#   1. Read GSEA results; extract leading-edge ENSG IDs from top immune pathway
#      (most negative NES → strongest immune suppression).
#   2. Map ENSG → gene symbol via DEA table (avoids Biomart calls).
#   3. Query STRINGdb (Homo sapiens, score ≥ 400) for PPI edges.
#   4. Build igraph; compute degree, betweenness, Louvain communities.
#   5. Four-panel ggraph figure:
#        A – Main network (FR layout, node colour = log2FC, shape = hub)
#        B – Community structure (KK layout)
#        C – Degree distribution
#        D – Hub-gene ranking bar chart
# ==============================================================================

# ── 1. Package installation & loading ─────────────────────────────────────────
pkgs_cran  <- c("igraph", "ggraph", "tidygraph", "ggplot2",
                "dplyr", "scales", "RColorBrewer", "patchwork",
                "ggrepel", "stringr")
pkgs_bioc  <- c("STRINGdb")

for (p in pkgs_cran) {
  if (!requireNamespace(p, quietly = TRUE))
    install.packages(p, repos = "https://cloud.r-project.org", quiet = TRUE)
}
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager", repos = "https://cloud.r-project.org", quiet = TRUE)
for (p in pkgs_bioc) {
  if (!requireNamespace(p, quietly = TRUE))
    BiocManager::install(p, ask = FALSE, quiet = TRUE)
}

suppressPackageStartupMessages({
  library(STRINGdb)
  library(igraph)
  library(ggraph)
  library(tidygraph)
  library(ggplot2)
  library(dplyr)
  library(scales)
  library(RColorBrewer)
  library(patchwork)
  library(ggrepel)
  library(stringr)
})
set.seed(42)  # Louvain + fr/kk layouts are stochastic

# ── 2. Working directory — always resolves to the script's own folder ─────────
args <- commandArgs(trailingOnly = FALSE)
file_match <- grep("--file=", args, value = TRUE)
if (length(file_match) > 0) {
  script_path <- sub("--file=", "", file_match[1])
  script_path <- gsub("~\\+~", " ", script_path)
  script_dir <- dirname(normalizePath(script_path))
} else {
  script_dir <- tryCatch(
    dirname(normalizePath(sys.frame(1)$ofile)),
    error = function(e)
      tryCatch(dirname(rstudioapi::getActiveDocumentContext()$path),
               error = function(e2) getwd())
  )
}
setwd(script_dir)
message("[ Working directory ] ", getwd())

# ── 3. Load data ───────────────────────────────────────────────────────────────
dea  <- read.csv("DEA_Full_Results_macOS.csv",  row.names = 1,
                 stringsAsFactors = FALSE)
gsea <- read.csv("GSEA_Results_Detailed.csv",   row.names = 1,
                 stringsAsFactors = FALSE)

colnames(dea) <- c("baseMean", "log2FC", "lfcSE", "stat",
                   "pvalue", "padj", "symbol")

gsea$NES       <- as.numeric(gsea$NES)
gsea$p.adjust  <- as.numeric(gsea$p.adjust)
gsea$setSize   <- as.numeric(gsea$setSize)

message(sprintf("[ DEA ] %d genes | %d with symbols",
                nrow(dea), sum(!is.na(dea$symbol) & dea$symbol != "")))
message(sprintf("[ GSEA ] %d pathways | %d FDR<0.05",
                nrow(gsea), sum(gsea$p.adjust < 0.05, na.rm = TRUE)))

# ── 4. Extract leading-edge ENSG IDs from top immune-suppressed pathway ────────
sig_neg <- gsea[!is.na(gsea$NES) & gsea$NES < 0 &
                !is.na(gsea$p.adjust) & gsea$p.adjust < 0.05, ]
top_pw  <- sig_neg[which.min(sig_neg$NES), ]

message(sprintf("[ Top pathway ] %s (NES = %.3f, FDR = %.2e)",
                top_pw$Description, top_pw$NES, top_pw$p.adjust))

raw_ids  <- unlist(strsplit(as.character(top_pw$core_enrichment), "/"))
ensg_ids <- trimws(raw_ids[nchar(trimws(raw_ids)) > 0])
message(sprintf("[ Leading-edge ENSG IDs ] %d", length(ensg_ids)))

# ── 5. Map ENSG → gene symbol via the DEA table ────────────────────────────────
# Row names are versioned (e.g. ENSG00000173757.12); strip version for matching
dea$ensg_base  <- sub("\\..*$", "", rownames(dea))
query_base     <- sub("\\..*$", "", ensg_ids)

sym_map <- dea[match(query_base, dea$ensg_base),
               c("ensg_base", "symbol", "log2FC", "padj")]
sym_map$ensg_query <- query_base
sym_map <- sym_map[
  !is.na(sym_map$symbol) &
  sym_map$symbol != "" &
  sym_map$symbol != "NA", ]
sym_map <- sym_map[!duplicated(sym_map$symbol), ]

gene_symbols <- sym_map$symbol
message(sprintf("[ Symbols mapped ] %d / %d",
                length(gene_symbols), length(ensg_ids)))

if (length(gene_symbols) < 5)
  stop("Too few symbols resolved. Check ENSG versioning in DEA file.")

# ── 6. STRINGdb: map genes & retrieve interactions ────────────────────────────
string_db <- STRINGdb$new(
  version         = "11.5",
  species         = 9606,     # Homo sapiens
  score_threshold = 400,
  input_directory = ""
)

gene_df  <- data.frame(gene = gene_symbols, stringsAsFactors = FALSE)
mapped   <- string_db$map(gene_df, "gene", removeUnmappedRows = TRUE)
message(sprintf("[ STRING-mapped proteins ] %d", nrow(mapped)))

interactions <- string_db$get_interactions(mapped$STRING_id)
message(sprintf("[ PPI edges retrieved ] %d", nrow(interactions)))

if (nrow(interactions) == 0)
  stop("No interactions found. Try lowering score_threshold.")

# ── 7. Build igraph ────────────────────────────────────────────────────────────
id2sym <- setNames(mapped$gene, mapped$STRING_id)

edges_df <- interactions %>%
  mutate(from_sym = id2sym[from],
         to_sym   = id2sym[to]) %>%
  filter(!is.na(from_sym), !is.na(to_sym)) %>%
  select(from   = from_sym,
         to     = to_sym,
         weight = combined_score)

g <- graph_from_data_frame(edges_df, directed = FALSE)
g <- simplify(g, remove.multiple = TRUE, remove.loops = TRUE)

# Node metrics
V(g)$degree      <- degree(g)
V(g)$betweenness <- betweenness(g, weights = NA, normalized = TRUE)  # STRING score is confidence, not distance
V(g)$closeness   <- closeness(g, weights = NA, normalized = TRUE)

# Hub: top 10 % by degree
deg_thr      <- quantile(V(g)$degree, 0.90)
V(g)$is_hub  <- V(g)$degree >= deg_thr

# Louvain community detection
comm         <- cluster_louvain(g)
V(g)$community <- membership(comm)

# Attach log2FC from DEA
fc_lut       <- setNames(sym_map$log2FC, sym_map$symbol)
V(g)$log2FC  <- unname(fc_lut[V(g)$name])
V(g)$log2FC[is.na(V(g)$log2FC)] <- 0

n_comm <- max(V(g)$community)
message(sprintf("[ igraph ] nodes=%d | edges=%d | communities=%d | hubs=%d",
                vcount(g), ecount(g), n_comm, sum(V(g)$is_hub)))

# Hub gene summary
hub_df <- data.frame(
  gene        = V(g)$name[V(g)$is_hub],
  degree      = V(g)$degree[V(g)$is_hub],
  betweenness = round(V(g)$betweenness[V(g)$is_hub], 4),
  log2FC      = round(V(g)$log2FC[V(g)$is_hub], 3)
) %>% arrange(desc(degree))
message("[ Hub genes ]")
print(hub_df)

# ── 8. ggraph visualisation ────────────────────────────────────────────────────
tg      <- as_tbl_graph(g)
n_show  <- min(n_comm, 9)
pal_set <- colorRampPalette(brewer.pal(n_show, "Set1"))(n_comm)

# ── Panel A: FR layout — node colour = log2FC, shape = hub ────────────────────
p_A <- ggraph(tg, layout = "fr") +
  geom_edge_link(aes(width = weight),
                 alpha = 0.18, colour = "grey55",
                 show.legend = FALSE) +
  geom_node_point(aes(
    size   = degree,
    colour = log2FC,
    shape  = factor(is_hub, levels = c(FALSE, TRUE),
                    labels = c("Gene", "Hub"))
  )) +
  geom_node_label(
    aes(label  = if_else(is_hub, name, ""),
        colour = log2FC),
    repel       = TRUE,
    size        = 3,
    fontface    = "bold",
    fill        = alpha("white", 0.75),
    label.size  = 0.15,
    max.overlaps = 25,
    show.legend = FALSE
  ) +
  scale_edge_width(range = c(0.15, 1.4)) +
  scale_size(range = c(1.5, 9), name = "Degree") +
  scale_colour_gradient2(
    low      = "#2166ac",
    mid      = "grey92",
    high     = "#d6604d",
    midpoint = 0,
    name     = "log₂FC",
    limits   = c(-6, 6),
    oob      = squish
  ) +
  scale_shape_manual(
    values = c("Gene" = 16, "Hub" = 17),
    name   = ""
  ) +
  labs(
    title    = paste0("A  PPI Network — Leading-edge genes"),
    subtitle = str_wrap(
      sprintf("Pathway: %s  |  NES = %.2f  |  FDR = %.1e",
              top_pw$Description, top_pw$NES, top_pw$p.adjust), 80),
    caption  = "Node colour: log₂FC (blue = down-reg, red = up-reg)  ·  Triangle = hub gene (degree ≥ 90th pctl)"
  ) +
  theme_graph(base_family = "sans") +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(size = 8.5, colour = "grey40"),
    plot.caption  = element_text(size = 7.5, colour = "grey55"),
    legend.text   = element_text(size = 8),
    legend.title  = element_text(size = 9, face = "bold"),
    plot.background = element_rect(fill = "#FAFAFA", colour = NA)
  )

# ── Panel B: KK layout — Louvain communities ──────────────────────────────────
p_B <- ggraph(tg, layout = "kk") +
  geom_edge_link(alpha = 0.12, colour = "grey65") +
  geom_node_point(aes(
    size   = degree,
    colour = factor(community)
  )) +
  geom_node_label(
    aes(label  = if_else(degree >= quantile(degree, 0.85), name, ""),
        colour = factor(community)),
    repel        = TRUE,
    size         = 2.7,
    fill         = alpha("white", 0.75),
    label.size   = 0.12,
    fontface     = "bold",
    max.overlaps = 20,
    show.legend  = FALSE
  ) +
  scale_size(range = c(1.5, 8), name = "Degree") +
  scale_colour_manual(values = pal_set, guide = "none") +
  labs(
    title    = "B  Community Structure (Louvain)",
    subtitle = sprintf("%d communities detected  |  Nodes = %d  |  Edges = %d",
                       n_comm, vcount(g), ecount(g))
  ) +
  theme_graph(base_family = "sans") +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(size = 8.5, colour = "grey40"),
    plot.background = element_rect(fill = "#FAFAFA", colour = NA)
  )

# ── Panel C: Degree distribution ──────────────────────────────────────────────
deg_df <- data.frame(
  degree = V(g)$degree,
  is_hub = V(g)$is_hub
)

p_C <- ggplot(deg_df, aes(x = degree, fill = is_hub)) +
  geom_histogram(binwidth = 1, colour = "white", linewidth = 0.25) +
  scale_fill_manual(
    values = c("FALSE" = "#74add1", "TRUE" = "#d73027"),
    labels = c("FALSE" = "Regular gene", "TRUE" = "Hub gene"),
    name   = ""
  ) +
  labs(
    title    = "C  Degree Distribution",
    subtitle = sprintf("Mean = %.1f  |  Max = %d  |  Hub threshold ≥ %g",
                       mean(deg_df$degree), max(deg_df$degree), deg_thr),
    x = "Node degree", y = "Count"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title      = element_text(face = "bold", size = 13),
    plot.subtitle   = element_text(size = 8.5, colour = "grey40"),
    legend.position = "top",
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = "#FAFAFA", colour = NA)
  )

# ── Panel D: Hub gene bar chart (degree + log2FC fill) ────────────────────────
centrality_all <- data.frame(
  gene        = V(g)$name,
  degree      = V(g)$degree,
  betweenness = V(g)$betweenness,
  log2FC      = V(g)$log2FC,
  is_hub      = V(g)$is_hub
)
top_n_plot <- centrality_all %>%
  arrange(desc(degree)) %>%
  head(20)

p_D <- ggplot(top_n_plot,
              aes(x = reorder(gene, degree), y = degree, fill = log2FC)) +
  geom_col(width = 0.72, colour = "white", linewidth = 0.3) +
  geom_text(aes(label = degree), hjust = -0.25, size = 3, colour = "grey30") +
  geom_point(data = top_n_plot %>% filter(is_hub),
             aes(x = reorder(gene, degree), y = degree + 0.3),
             colour = "#d73027", shape = 17, size = 2.5, inherit.aes = FALSE) +
  coord_flip() +
  scale_fill_gradient2(
    low      = "#2166ac",
    mid      = "grey92",
    high     = "#d6604d",
    midpoint = 0,
    name     = "log₂FC",
    limits   = c(-6, 6),
    oob      = squish
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "D  Top-20 Genes by Node Degree",
    subtitle = "▲ marks hub genes (top 10 % by degree)",
    x = NULL, y = "Node degree"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title      = element_text(face = "bold", size = 13),
    plot.subtitle   = element_text(size = 8.5, colour = "grey40"),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.background = element_rect(fill = "#FAFAFA", colour = NA)
  )

# ── Assemble 4-panel layout with patchwork ────────────────────────────────────
combined_fig <- (p_A | p_B) / (p_C | p_D) +
  plot_annotation(
    title    = "Protein–Protein Interaction Network Analysis",
    subtitle = sprintf(
      "Leading-edge genes: %s  ·  STRINGdb v11.5 (Homo sapiens, combined score ≥ 400)",
      str_trunc(top_pw$Description, 70)),
    caption  = "Data: DEA_Full_Results_macOS.csv + GSEA_Results_Detailed.csv",
    theme = theme(
      plot.title    = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 10, colour = "grey40"),
      plot.caption  = element_text(size = 8,  colour = "grey60"),
      plot.background = element_rect(fill = "#FAFAFA", colour = NA)
    )
  )

# ── 9. Save outputs ────────────────────────────────────────────────────────────
ggsave("PPI_Network_Plot.png",
       combined_fig, width = 18, height = 14, dpi = 180,
       bg = "#FAFAFA")

ggsave("PPI_Network_LeadingEdge.pdf",
       combined_fig, width = 18, height = 14,
       device = cairo_pdf)

# Centrality metrics table
centrality_out <- centrality_all %>%
  arrange(desc(degree)) %>%
  mutate(betweenness = round(betweenness, 5),
         log2FC      = round(log2FC, 3))
write.csv(centrality_out, "PPI_centrality_metrics.csv", row.names = FALSE)

# Hub gene summary
write.csv(hub_df, "PPI_hub_genes.csv", row.names = FALSE)

# ── 10. Console summary ────────────────────────────────────────────────────────
message(rep("─", 60))
message("  PPI ANALYSIS COMPLETE")
message(rep("─", 60))
message(sprintf("  Input genes (symbols)  : %d", length(gene_symbols)))
message(sprintf("  STRING-mapped proteins : %d", nrow(mapped)))
message(sprintf("  PPI nodes (in graph)   : %d", vcount(g)))
message(sprintf("  PPI edges              : %d", ecount(g)))
message(sprintf("  Louvain communities    : %d", n_comm))
message(sprintf("  Hub genes (≥90th pctl) : %d", sum(V(g)$is_hub)))
message(sprintf("  Top hub               : %s (degree=%d)",
                hub_df$gene[1], hub_df$degree[1]))
message("  Outputs:")
message(sprintf("    PPI_Network_Plot.png"))
message(sprintf("    PPI_Network_LeadingEdge.pdf"))
message(sprintf("    PPI_centrality_metrics.csv"))
message(sprintf("    PPI_hub_genes.csv"))
message(rep("─", 60))
