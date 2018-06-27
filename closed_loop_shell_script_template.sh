#!/bin/sh
echo *******NEW SIMULATION******
# Set hyperparamaters
sim_mode=lo
gamma=1
epsilon=0.8
beta=$(bc <<< "scale=5; 0.1/$epsilon^10")
seed=0

# Display hyperparameters
echo sim_mode=$sim_mode gamma=$(echo "$gamma"|bc) epsilon=$(echo "$epsilon"|bc) beta=$(echo "$beta"|bc) seed=$seed

# Loop through each round
for i in 0 1 2 3 4 5 6 7 8 9
do
    python closed_loop_oed.py --round_idx=$i --sim_mode=$sim_mode --gamma=$gamma --epsilon=$epsilon --beta=$beta --seed=$seed
    python generate_predictions.py --round_idx=$i --sim_mode=$sim_mode --seed=$seed
done
