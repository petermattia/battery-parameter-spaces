import os, sys

run_id = sys.argv[1]
num_seeds = 2000
current_dir = 'oed_vs_random_validation'

script_dir = os.path.join(os.getcwd(), current_dir, 'bash_scripts', run_id)
all_script_files = []
for hparam_idx in os.listdir(script_dir):
	subscript_dir = os.path.join(script_dir, hparam_idx)
	if os.path.isdir(subscript_dir):
		print(hparam_idx, end=' ')
		for seed in range(num_seeds):
			if seed%100 == 0:
				# print(seed, end=' ')
				script_file = os.path.join(subscript_dir, 'seed_' + str(seed) +'.sh')
				all_script_files.append(script_file)

f = open(os.path.join(os.getcwd(), current_dir, 'all_scripts_template.sh'),'r')
text = f.read()
f.close()

all_script_cmds = ['srun --partition=atlas bash ' + script_file + ' &' for script_file in all_script_files]
slurm_text = text.replace('srun_proxy', '\n'.join(all_script_cmds))
slurm_file = os.path.join(os.getcwd(), current_dir, 'bash_scripts', run_id, 'all_scripts.sh')
with open(slurm_file, 'w') as outfile:
	outfile.write(slurm_text)
