#! /usr/bin/env bash

R --no-save --silent <<-EOF
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
	# BiocManager::install(version = "3.11")
	BiocManager::install("edgeR")
EOF
