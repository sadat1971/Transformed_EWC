# Transformed_EWC
The repo is a temporary placeholder for the codes and data used in the paper:#7   Transformed-EWC: A Regularization Technique for Continual Learning of Misinformation Detection by Mitigating Catastrophic Forgetting. It will be updated over time

To run:

1. Shpwing the CF effect (no-reg):

Note: use anything in Task1, Task2 and Task3 based on your requirement

`python3 Perform_training.py --Task1 news --Task2 tweets --Task3 revs --batch_size 32 --epochs1 50 --epochs2 50 --epochs3 50 --learning_rate 5e-5 --hidden_size1 32 --hidden_size2 32 --device cuda:0 --dropout1 0 --dropout2 0 --Task_description Triple --mode no-reg`

2. dropout:

`python3 Perform_training.py --Task1 news --Task2 tweets --Task3 revs --batch_size 32 --epochs1 50 --epochs2 50 --epochs3 50 --learning_rate 5e-5 --hidden_size1 32 --hidden_size2 32 --device cuda:1 --dropout1 .3 --dropout2 .3 --use_L2_or_dropout dropout --Task_description Triple --mode dropout`

3. EWC:

`python3 Perform_training.py --Task1 news --Task2 tweets --Task3 revs --batch_size 32 --epochs1 50 --epochs2 50 --epochs3 50 --learning_rate 5e-5 --hidden_size1 32 --hidden_size2 32 --device cuda:1 --dropout1 .3 --dropout2 .3 --use_L2_or_dropout dropout --Task_description Triple --mode EWC --Lambda_optimization no`

4. TEWC:

`python3 Perform_training.py --Task1 news --Task2 tweets --Task3 revs --batch_size 32 --epochs1 50 --epochs2 50 --epochs3 50 --learning_rate 5e-5 --hidden_size1 32 --hidden_size2 32 --device cuda:1 --dropout1 .3 --dropout2 .3 --use_L2_or_dropout dropout --Task_description Triple --mode TEWC --Lambda_optimization no`

