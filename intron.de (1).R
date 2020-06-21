library(edgeR)
library(gplots)

#  read counts for each sample from CNT-matrix

cntKR=read.table(file="matched.csv", header=T, row.names=1, check.names=F)
dgList.KR <- DGEList(counts=cntKR, genes=rownames(cntKR))

numSamples=(dim(cntKR)[2])/2

cpm.KR=cpm(dgList.KR)
summary(cpm.KR)

min_cpm=1

cnt.KR.check=cpm.KR>min_cpm
head(cnt.KR.check)
keep.KR=which(rowSums(cnt.KR.check) >=10)
head(keep.KR)
dgList.KR=dgList.KR[keep.KR,]
summary(cpm(dgList.KR))

# calculate normalization factor

dgList.KR=calcNormFactors(dgList.KR, method="TMM")
mds.s1=plotMDS(dgList.KR[1:10000,], xlim=c(-3,3))
pch_type=c(rep(15,numSamples), rep(16,numSamples))
col_type=c(rep("red",numSamples), rep("blue",numSamples))
plotMDS(dgList.KR[1:10000,], xlim=c(-3,3), col=col_type, pch=pch_type, cex=1, xlab="Dim 1", ylab="Dim 2")

# generate design matrix

type=c(rep("normal",numSamples), rep("tumor", numSamples))
target=data.frame(Type=type)
rownames(target)=colnames(dgList.KR)

group=factor(target$Type)
groupName=c("normal", "tumor")
design=model.matrix(~0+group, data=target)
colnames(design)=c("normal", "tumor")
design

# estimate dispension factors

dgList.KR = estimateDisp(dgList.KR, design)
plotBCV(dgList.KR)

# pairwise DEG with Ctrl

fit.KR <- glmQLFit(dgList.KR, design)
qlf.KR <- glmQLFTest(fit.KR, contrast=c(-1,1))
topTags(qlf.KR)






