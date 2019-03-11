import itertools
import numpy as np
import os, shutil

# real params


num_policies = 8
num_seeds = 2000
pop_budget = 5
run_id = 'mar10'
logdir = './logs/' + run_id
if not os.path.exists(logdir):
	os.mkdir(logdir)

policy_file = 'paper_plot/final_results.csv'

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

script_dir = os.path.join(os.getcwd(), 'paper_plot', 'bash_scripts', run_id)

if os.path.exists(script_dir):
	shutil.rmtree(script_dir)

os.mkdir(script_dir)


all_script_files = []
for seed in range(num_seeds):
	train_policy_list = np.random.permutation(num_policies).tolist()
	cmd_prefix = 'python paper_plot/closed_loop_oed.py'\
					+ ' --dump'\
					+ ' --policy_file ' + policy_file\
					+ ' --logdir ' + logdir\
					+ ' --exp_id ' + str(current_exp_id)\
					+ ' --max_budget ' +  str(max_budget)\
					+ ' --pop_budget ' +  str(pop_budget)\
					+ ' --train_policy_idx ' + ' '.join(map(str, train_policy_list)) \
					+ ' --seed ' + str(seed)
	if seed%10 == 0:
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
f = open(os.path.join(os.getcwd(), 'paper_plot', 'all_scripts_template.sh'),'r')
text = f.read()
f.close()

all_script_cmds = ['srun --partition=atlas bash ' + script_file + ' &' for script_file in all_script_files]
slurm_text = text.replace('srun_proxy', '\n'.join(all_script_cmds))
slurm_file = os.path.join(os.getcwd(), 'paper_plot', 'bash_scripts', run_id, 'current_scripts_template.sh')
with open(slurm_file, 'w') as outfile:
	outfile.write(slurm_text)


print('total exp folder', current_exp_id-1)
print('total dumped cmds', num_cmds)




