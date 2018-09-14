### Hyperparameter optimization notes

- List of hyperparameters (from closed_loop_oed_hyperparameters.py)

beta_list = [0.2,0.5,1.0,2.0,5.0,10.0]

gamma_list = [0.1,0.3,1.0,3.0,10.0,30.0,100.0]

epsilon_list = [0.5,0.6,0.7,0.8,0.9]

seed_list = [0,1,2,3,4,5,6,7,8,9]

- Run on 224-policy space
- Number of rounds = min from simulator (6) + 2 "bonus" = `8`
- Results: `init_beta = 5.0`, `gamma = 1.0`, `epsilon = 0.5`
- `seed=0`
- In plots, gamma is denoted by color in the legend