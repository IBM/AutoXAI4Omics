#! /usr/bin/env bash

R --no-save --silent <<-EOF
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", ask = FALSE)
	BiocManager::install(version = "3.12", ask = FALSE)
	BiocManager::install("edgeR", ask = FALSE)
EOF
