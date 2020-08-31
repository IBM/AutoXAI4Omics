#! /usr/bin/R

#Load library
library(edgeR)

#Read in command line options
args <- commandArgs()

#Read in inout file
y <- read.csv(args[5], row.names="gene", header= T,sep = ",")
print("Gene number:")
dim(y)

#Filter genes 
filter1=as.integer(args[6])
filter2=as.integer(args[7])
countCheck <- y > filter1
keep <- which(rowSums(countCheck) >= filter2) 
y <- y[keep,]
dim(y)

#Transform matrix for ML so samples = rows and genes = columns(or features)
ty <- t(y)
print("Sample number:")
dim(ty)

#Filter samples
filter4=as.integer(args[8])
means <- rowMeans(ty > 0) #Finds the proportion of genes meeting the criteria (cov > 0) for each of the samples (rows)
mean_for_all <- mean(means) #Averages across the samples
sd_for_all <- sd(means)	#ST dev acorss the samples
print("Average proportion of genes meeting criteria across the samples:")
print(mean_for_all)
print("Standard Deviation of genes meeting criteria across the samples:")
print(sd_for_all)

STD <- (filter4 * sd_for_all)
print("Standard Deviation x user value:")
print(STD)
print("Range (mean +/- StDev) for filtering samples to keep:")
print(range((mean_for_all - STD),(mean_for_all + STD)))

keep_feature <- which((rowMeans(ty > 0) >= (mean_for_all - STD)) & (rowMeans(ty > 0) <= (mean_for_all + STD)))
ty <- ty[keep_feature, ]
print("Sample number remaining:")
dim(ty)


write.csv(ty, file=args[9])

#dev.off()

