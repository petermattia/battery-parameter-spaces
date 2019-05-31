
import os, sys
import numpy as np
import pickle
from functools import partial
import json
import concurrent.futures
from collections import defaultdict

np.random.seed(0)

run_id = sys.argv[1]
logdir = './logs/' + run_id

with open(os.path.join(logdir, 'aggregated_results.pkl'), 'rb') as infile:
    results_list = pickle.load(infile)
    print(len(results_list))

all_best_hparams = []
complete_best_hparams = []
for round_idx, round_data in enumerate(results_list):
    print('Round', round_idx)
    best_loss = np.inf
    for hparam_config, hparam_losses in round_data.items():
        avg_hparam_loss = np.mean(np.array(hparam_losses))
        # print(hparam_config, avg_hparam_loss)
        # print(hparam_losses)
        if avg_hparam_loss < best_loss:
            best_hparams = [hparam_config]
            best_loss = avg_hparam_loss
        elif avg_hparam_loss == best_loss:
            best_hparams.append(hparam_config)
    random_tiebreak = best_hparams[np.random.choice(len(best_hparams))]
    all_best_hparams.append(random_tiebreak)
    complete_best_hparams.append(best_hparams)

    print('Summary', len(best_hparams), best_loss, random_tiebreak, best_hparams)
    print()
print(set(all_best_hparams[1:]), len(set(all_best_hparams[1:])))

run_id_2 = sys.argv[2]
logdir_2 = './logs/' + run_id_2

def dump_commands():
    for best_idx, best_hparams in enumerate(set(all_best_hparams[1:])):
        cmd = 'python ./oed_vs_random_validation/dump_commands_best.py '\
                    + run_id_2 + '/' + str(best_idx) + ' '\
                    + str(best_hparams[0]) + ' '\
                    + str(best_hparams[1]) + ' '\
                    + str(best_hparams[2]) + ' '\
                    + early_pred + ' '\
                    + apply_correction
        print(cmd)
        os.system(cmd)

if sys.argv[3].lower() == 'true':
    # dump commands based on best hparams
    early_pred = sys.argv[4]
    apply_correction = sys.argv[5]
    dump_commands()
else:

    # aggregate results based on best hparams
    # def aggregate_results():
        

    total_exp_folders = 2000
    print(total_exp_folders, flush=True)

    pop_budget = 5
    num_policies = 9
    max_budget = pop_budget*num_policies
    end_idx = 3
    results_list = []
    perf_list = []
    experiment_cycles_list = []
    for _ in range(max_budget+1):
        results_list.append(defaultdict(list))
        perf_list.append(defaultdict(list))
        experiment_cycles_list.append(defaultdict(list))

    def get_all_data(logdir_hparam, exp_id):
        if exp_id % 100 == 0:
            print(exp_id, end=' ', flush=True)

        all_round_data = []
        # all_true_lifetimes = []
        for round_idx in range(max_budget+1):
            with open(os.path.join(logdir_hparam, str(exp_id), 'round_' + str(round_idx) +  '.txt')) as infile:
                lines = infile.readlines()
            true_lifetimes = np.array([float(lifetime) for lifetime in lines[0].rstrip().split("\t")])
            pred_rankings = [int(rank) for rank in lines[2].rstrip().split("\t")]
            oed_loss = np.amax(true_lifetimes)-true_lifetimes[pred_rankings][0]
            oed_perf = true_lifetimes[pred_rankings][0]
            all_round_data.append((oed_loss, oed_perf))
            # all_true_lifetimes.append(np.amax(true_lifetimes))

        experiment_cycles = [0]
        with open(os.path.join(logdir_hparam, str(exp_id),'log.txt')) as infile:
            lines = infile.readlines()
        for line in lines:
            tokens = line.rstrip().split(' ')
            if tokens[0] == 'Parameters':
                experiment_cycles.append(float(tokens[-2]))
        experiment_cycles = np.cumsum(np.array(experiment_cycles))
            
        return all_round_data, experiment_cycles

    tested_list = []
    for best_idx in range(len(set(all_best_hparams[1:]))):
        logdir_hparam = logdir_2 + '/' + str(best_idx)
        round_get_all_data = partial(get_all_data, logdir_hparam)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     for all_round_data in executor.map(round_get_all_data, range(1, total_exp_folders+1)):
        #         # print(all_round_data)
        #         for round_idx, round_data in enumerate(all_round_data):
        #             # print(round_data)
        #             # import pdb
        #             # pdb.set_trace()
        #             results_list[round_idx][(best_hparams[0], best_hparams[1], best_hparams[2])].append(round_data[0])
        #             perf_list[round_idx][(best_hparams[0], best_hparams[1], best_hparams[2])].append(round_data[1])
        #             # print(np.mean(np.array(results_list[round_idx][(best_hparams[0], best_hparams[1], best_hparams[2])])))
        #             # print(np.mean(np.array(perf_list[round_idx][(best_hparams[0], best_hparams[1], best_hparams[2])])))
        with open(os.path.join(logdir_hparam, '1', 'config.json')) as infile:
            config = json.load(infile)
        beta = config["init_beta"]
        gamma = config["gamma"]
        epsilon = config["epsilon"]
        print(logdir_hparam, beta, gamma, epsilon)

        # for exp_id in range(1, total_exp_folders+1):
            # all_round_data = round_get_all_data(exp_id)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for all_round_data, experiment_cycles in executor.map(round_get_all_data, range(1, total_exp_folders+1)):
                for round_idx, round_data in enumerate(all_round_data):
                    results_list[round_idx][(beta, gamma, epsilon)].append(round_data[0])
                    perf_list[round_idx][(beta, gamma, epsilon)].append(round_data[1])
                    experiment_cycles_list[round_idx][(beta, gamma, epsilon)].append(experiment_cycles[round_idx])
        tested_list.append((beta, gamma, epsilon))

    best_losses = [] # list of lists
    best_perfs = []
    best_experiment_cycles = []
    for best_idx, tied_best_hparams in enumerate(complete_best_hparams):
        if best_idx == 0 or best_idx>max_budget: # ignoring round 0 coz we didn't run it (random)
            continue
        print(best_idx)
        for best_hparams in tied_best_hparams:
            if best_hparams in tested_list:
                print(best_hparams, np.mean(np.array(perf_list[best_idx][(best_hparams[0], best_hparams[1], best_hparams[2])])), np.mean(np.array(results_list[best_idx][(best_hparams[0], best_hparams[1], best_hparams[2])])))
                best_perfs.append(perf_list[best_idx][(best_hparams[0], best_hparams[1], best_hparams[2])])
                best_losses.append(results_list[best_idx][(best_hparams[0], best_hparams[1], best_hparams[2])])
                best_experiment_cycles.append(experiment_cycles_list[best_idx][(best_hparams[0], best_hparams[1], best_hparams[2])])
                break
        print()

    with open(os.path.join(logdir_2, 'best_experiment_cycles.pkl'),'wb') as outfile:
        pickle.dump(best_experiment_cycles, outfile)

    with open(os.path.join(logdir_2, 'best_oed_losses.pkl'), 'wb') as outfile:
      pickle.dump(best_losses, outfile)

    with open(os.path.join(logdir_2, 'best_oed_lifetimes.pkl'), 'wb') as outfile:
      pickle.dump(best_perfs, outfile)


# aggregate_results()