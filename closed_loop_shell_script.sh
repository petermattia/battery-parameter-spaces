#!/bin/sh
echo *******NEW SIMULATION******
# Set hyperparamaters
beta=5.0
gamma=1
epsilon=0.5
seed=0

# Display hyperparameters
echo beta=$(echo "$beta"|bc) gamma=$(echo "$gamma"|bc) epsilon=$(echo "$epsilon"|bc) seed=$seed

# Loop through each round
for i in 0 1 2 3 4 5 6 7
do
    python closed_loop_oed.py --round_idx=$i --init_beta=$beta --gamma=$gamma --epsilon=$epsilon --seed=$seed
    python generate_predictions.py --round_idx=$i --seed=$seed
done

# One more round
python closed_loop_oed.py --round_idx=8 --init_beta=$beta --gamma=$gamma --epsilon=$epsilon --seed=$seed