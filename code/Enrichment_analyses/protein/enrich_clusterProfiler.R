
library(clusterProfiler)
library(org.Hs.eg.db)
require(dplyr)
require(ggplot)

pro_gene <- read.csv(file = '/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/protein/pro_rmse_largerthan_Q3.csv')
GO <- enrichGO(gene=pro_gene$symbol %>% unique %>% as.character() ,keyType='SYMBOL',OrgDb=org.Hs.eg.db,ont='BP')
#dotplot(GO,showCategory=10) + theme(axis.text.y = element_text(size=15))
#ggsave(filename = '/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/protein/protein.jpg',width = 15,height=10)

View(GO@result)
## ============================== based on p-value
compute.ratio <- function(s) {
  v <- strsplit(s,split='/') %>% unlist  
  as.numeric(v[1]) / as.numeric(v[2])
  
}

GO.enrichment.dotplot <- function(BP.df) {
  #BP.df$Description   <- paste(BP.df$Description,sprintf('(%d)',BP.df$Count), sep =' ' )
  BP.df$x             <- sapply(BP.df$GeneRatio %>% as.character,compute.ratio)
  BP.df               <- BP.df[order(BP.df$x,decreasing = TRUE),]  
  BP.df$y             <- factor(BP.df$Description,levels = BP.df$Description %>% rev)
  #mid                 <-mean(-1 * log10(BP.df$pvalue))
  p <- ggplot(BP.df) +
    geom_point( aes(x=x, y=y,size=Count,color= -1 * log10(pvalue) ) )  + scale_color_gradient(low="blue", high="red") + # weired trick, but working!
    
    theme_bw(base_size = 10) +
    
    theme(axis.title   = element_text( size=15, face="bold"),
          axis.text.y  = element_text( size=20, face="bold"),
          axis.text.x  = element_text(size=15, face="bold"),
          plot.margin = margin(0.05, 0.05, 0.05, 0.05, "cm"),
          axis.line.x = element_line(colour = "black",size = 2),
          axis.line.y = element_line(colour = "black",size = 2),
          legend.text = element_text(size=15, face="bold"),
          legend.title = element_text(size=15, face="bold") , aspect.ratio=18/6 )  + xlab('Gene Ratio') + ylab('') 
  p
}

GO.enrichment.dotplot(GO@result[1:10,])
ggsave(filename = '/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/protein/pro_Q1.jpg', width = 20,height=8)
ggsave(filename = '/Users/shanjuyeh/Desktop/Project/GeneExp_prediction/code/enrichment_analyses/protein/pro_Q3.jpg', width = 20,height=8)
