#!/bin/bash                                                                    
#SBATCH --partition=atlas --qos=normal                                         
#SBATCH --time=96:00:00                                                        
#SBATCH --nodes=1                                                              
#SBATCH --cpus-per-task=4                                                      
#SBATCH --mem=2G                                                                                                                    
                                                                               
#SBATCH --job-name="sample"                                                    
#SBATCH --output=sample-%j.out                                                 
                                                                               
# only use the following if you want email notification                        
#SBATCH --mail-user=adityag@cs.stanford.edu                                      
#SBATCH --mail-type=ALL                                                        
                                                                               
# list out some useful information                                             
echo "SLURM_JOBID="$SLURM_JOBID                                                
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST                                  
echo "SLURM_NNODES"=$SLURM_NNODES                                              
echo "SLURMTMPDIR="$SLURMTMPDIR                                                
echo "working directory = "$SLURM_SUBMIT_DIR                                   
                                                                               
# sample job                                                                   
NPROCS=`srun --nodes=${SLURM_NNODES} zsh -c 'hostname' |wc -l`                 
echo NPROCS=$NPROCS                                                            
                                                                               
# can try the following to list out which GPU you have access to               
srun_proxy          

# done                                                                         
echo "Done"                                                                    
