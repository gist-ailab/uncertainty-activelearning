BASELINE OF ACTIVE LEARNING

We provide software of baseline of active learning. We only provide confident, entropy based query algorithm. You can freely modify the code for your experiment.

HOW TO RUN

python main.py

with parser you can modify your data_path, save_path, epoch(Training epoch of each cycle), episode(total episode in AL), seed, gpu number, dataset, query_algorithm, addendum(num of data size you will labeling each cycle), batch_size

Provide Method 
random : query data randomly
high confidence : query data that current model high confiently predicted
low confidence : low confiently predicted
balance : query balance confidence data according to ranking