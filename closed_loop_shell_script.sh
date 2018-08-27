#!/bin/sh
echo *******NEW SIMULATION******
# Set hyperparamaters
beta=10.0
gamma=100.0
epsilon=0.9
seed=9

# Display hyperparameters
echo beta=$(echo "$beta"|bc) gamma=$(echo "$gamma"|bc) epsilon=$(echo "$epsilon"|bc) seed=$seed

# Loop through each round
for i in 0 1 2 3 4 5 6 7 8 9
do
    python closed_loop_oed.py --round_idx=$i --init_beta=$beta --gamma=$gamma --epsilon=$epsilon --seed=$seed
    python generate_predictions.py --round_idx=$i --seed=$seed
done

# One more round
python closed_loop_oed.py --round_idx=10 --init_beta=$beta --gamma=$gamma --epsilon=$epsilon --seed=$seed