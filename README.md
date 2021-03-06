# BitBucket
Link Prediction for Knowledge Graphs Based on Extended Relational Graph Attention Networks

###Dataset
- Freebase: FB15k-237
- Wordnet: WN18RR
- Nell: NELL-995
- Kinship: kinship
- UMLS: umls

###Running Environment: pytorch36 +python36

###Reproducing results
Dataset  UMLS
$python36   main.py 
--data ./data/UMLS/ 
--epochs_gat 1000 
--epochs_conv 300 
--weight_decay_gat 0.00001 
--margin 1 
--out_channels 50 
--drop_conv 0.3 
--weight_decay_conv 0.000001 
--batch_size_conv 256 
--output_folder ./checkpoints/umls/out/

Dataset  WN18RR
$python36   main.py 
--data ./data/WN18RR/ 
--epochs_gat 2000 
--epochs_conv 260 
--weight_decay_gat 0.00001 
--margin 1 
--out_channels 50 
--drop_conv 0.5 
--weight_decay_conv 0.000001 
--batch_size_conv 256 
--output_folder ./checkpoints/wn18rr/out/

###Training  Parameters
drop_GAT = {0.1,0.3,0.5}
batch_size_conv ={128.256,512}
