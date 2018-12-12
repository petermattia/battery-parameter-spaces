#!/bin/sh
echo *******NEW SIMULATION******
# Set hyperparamaters
beta=1
gamma=1
epsilon=0.8
seed=0
bsize=1

# Display hyperparameters
echo beta=$(echo "$beta"|bc) gamma=$(echo "$gamma"|bc) epsilon=$(echo "$epsilon"|bc) seed=$seed

# Loop through each round
for i in 0 1 2
do
    python baseline/closed_loop_oed.py --round_idx=$i --init_beta=$beta --gamma=$gamma --epsilon=$epsilon --seed=$seed --bsize=$bsize
    python baseline/generate_predictions.py --round_idx=$i
done

# One more round
python baseline/closed_loop_oed.py --round_idx=3 --init_beta=$beta --gamma=$gamma --epsilon=$epsilon --seed=$seed --bsize=$bsize