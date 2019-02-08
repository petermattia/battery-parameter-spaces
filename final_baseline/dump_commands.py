import itertools
import numpy as np
import os

# real params
beta_list = [0.2,0.5,1.0,2.0,5.0,10.0,10000.0]
gamma_list = [0.00001, 0.1,0.3,1.0,3.0,10.0,30.0,100.0]
epsilon_list = [0.5,0.6,0.7,0.8,0.9,1.0]

num_policies = 6
num_train_policies = 3
num_seeds = 10
pop_budget = 5
run_id = 'feb6'
logdir = './logs/' + run_id
if not os.path.exists(logdir):
	os.mkdir(logdir)

policy_file = 'final_baseline/predictions_3and3.csv'

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
max_budget = pop_budget*num_train_policies

print('total expected cmds', len(list(itertools.permutations(policy_list, num_train_policies)))*num_seeds*len(list(itertools.product(beta_list, gamma_list, epsilon_list)))*(max_budget+1))
current_exp_id = 1
num_cmds = 0

script_dir = os.path.join(os.getcwd(), 'final_baseline', 'bash_scripts', run_id)

if not os.path.exists(script_dir):
	os.mkdir(script_dir)

f = open(os.path.join(os.getcwd(), 'final_baseline', 'all_scripts_template.sh'),'r')
text = f.read()
f.close()

all_script_files = []
for seed in range(num_seeds):
	print(seed)
	# script_file = os.path.join(script_dir, 'seed_' + str(seed) +'.sh')
	# with open(script_file, 'w') as outfile:
	for train_policy_list in itertools.permutations(policy_list, num_train_policies):
		script_file = os.path.join(script_dir, 'seed_' + str(seed) + '_policy_list_' + ''.join(map(str, train_policy_list)) +'.sh')
		with open(script_file, 'w') as outfile:
			for beta, gamma, epsilon in itertools.product(beta_list, gamma_list, epsilon_list):
				cmd_prefix = 'python final_baseline/closed_loop_oed.py'\
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
				for round_idx in range(max_budget+1):
					cmd = cmd_prefix + ' --round_idx ' + str(round_idx) + '\n'
					num_cmds += 1
					# print(cmd)
					outfile.write(cmd)
				current_exp_id += 1
		all_script_files.append(script_file)

all_script_cmds = ['srun --partition=atlas bash ' + script_file + ' &' for script_file in all_script_files]
slurm_text = text.replace('srun_proxy', '\n'.join(all_script_cmds))
slurm_file = os.path.join(os.getcwd(), 'final_baseline', 'bash_scripts', run_id, 'current_scripts_template.sh')
with open(slurm_file, 'w') as outfile:
	outfile.write(slurm_text)


print('total exp folder', current_exp_id-1)
print('total dumped cmds', num_cmds)




