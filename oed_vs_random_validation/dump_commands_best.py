import itertools
import numpy as np
import os, shutil, sys

# real params


num_policies = 9
num_seeds = 2000
pop_budget = 5
run_id = sys.argv[1]
logdir = './logs/' + run_id
if not os.path.exists(logdir):
    os.makedirs(logdir)
beta = sys.argv[2]
gamma = sys.argv[3]
epsilon = sys.argv[4]

early_pred = sys.argv[5]
apply_correction = sys.argv[6]

current_dir = 'oed_vs_random_validation'
if early_pred.lower() == 'true':
    if apply_correction.lower() == 'true':
        policy_file = os.path.join(current_dir, 'predictions_debiased.csv')
    else:
        policy_file = os.path.join(current_dir, 'predictions.csv')
else:
    policy_file = os.path.join(current_dir, 'final_results.csv')


# dummy set of params, comment out block for final dumping
# beta_list = [0.2]
# gamma_list = [0.1]
# epsilon_list = [0.5,0.6]
# num_policies = 6
# num_train_policies = 3
# num_seeds = 1
# pop_budget = 3
####

policy_list = np.arange(num_policies).tolist()
max_budget = pop_budget*num_policies

current_exp_id = 1
num_cmds = 0

script_dir = os.path.join(os.getcwd(), current_dir, 'bash_scripts', run_id)

if os.path.exists(script_dir):
    shutil.rmtree(script_dir)

os.makedirs(script_dir)


all_script_files = []
for seed in range(num_seeds):
    train_policy_list = np.random.permutation(num_policies).tolist()
    cmd_prefix = 'python ' + current_dir + '/closed_loop_oed_best.py'\
                    + ' --dump'\
                    + ' --policy_file ' + policy_file\
                    + ' --logdir ' + logdir\
                    + ' --exp_id ' + str(current_exp_id)\
                    + ' --max_budget ' +  str(max_budget)\
                    + ' --pop_budget ' +  str(pop_budget)\
                    + ' --train_policy_idx ' + ' '.join(map(str, train_policy_list)) \
                    + ' --seed ' + str(seed)\
                    + ' --init_beta ' + str(beta)\
                    + ' --gamma ' + str(gamma)\
                    + ' --epsilon '  + str(epsilon)
    if early_pred.lower() == 'true':
        cmd_prefix += ' --early_pred'
    if apply_correction.lower() == 'true':
        cmd_prefix += ' --apply_correction'
    if seed%100 == 0:
        print(seed, end=' ')
        script_file = os.path.join(script_dir, 'seed_' + str(seed) +'.sh')
        all_script_files.append(script_file)

    with open(script_file, 'a') as outfile:
        for round_idx in range(max_budget+1):
            cmd = cmd_prefix + ' --round_idx ' + str(round_idx) + '\n'
            num_cmds += 1
            # print(cmd)
            outfile.write(cmd)
    current_exp_id += 1
# f = open(os.path.join(os.getcwd(), current_dir, 'all_scripts_template.sh'),'r')
# text = f.read()
# f.close()

# all_script_cmds = ['srun --partition=atlas bash ' + script_file + ' &' for script_file in all_script_files]
# slurm_text = text.replace('srun_proxy', '\n'.join(all_script_cmds))
# slurm_file = os.path.join(os.getcwd(), current_dir, 'bash_scripts', run_id, 'current_scripts_template.sh')
# with open(slurm_file, 'w') as outfile:
#   outfile.write(slurm_text)


print('total exp folder', current_exp_id-1)
print('total dumped cmds', num_cmds)




