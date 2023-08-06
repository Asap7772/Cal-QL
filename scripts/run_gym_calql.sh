export D4RL_SUPPRESS_IMPORT_ERROR=1
# export WANDB_DISABLED=True

exp_num=0
envs=(hopper-random-v2 hopper-medium-v2 hopper-medium-replay-v2 walker2d-random-v2 walker2d-medium-v2 walker2d-medium-replay-v2)
cql_min_q_weights=(20 1 5 10 5 5)
online_min_q_weights=(5 1)
seeds=(24 42)
gpus=(0 1 2 3 4 5 6 7)
num_max=24
max_online_env_steps=1e6

for env in ${envs[@]}; do
for online_min_q_weight in ${online_min_q_weights[@]}; do
for seed in ${seeds[@]}; do
    cql_min_q_weight=${cql_min_q_weights[$exp_num]}
    gpu=${gpus[$exp_num % ${#gpus[@]}]}
    export CUDA_VISIBLE_DEVICES=$gpu

    echo "Running experiment $exp_num on GPU $gpu: $env, seed=$seed, cql_min_q_weight=$cql_min_q_weight, online_min_q_weight=$online_min_q_weight"
    
    now=$(date +"%Y%m%d_%H%M%S")
    command="XLA_PYTHON_CLIENT_PREALLOCATE=false python -m JaxCQL.conservative_sac_main \
    --env=$env \
    --logging.online \
    --seed=$seed \
    --logging.project=Cal-QL-gym \
    --cql_min_q_weight=$cql_min_q_weight \
    --cql_min_q_weight_online=$online_min_q_weight \
    --policy_arch=256-256 \
    --qf_arch=256-256 \
    --offline_eval_every_n_epoch=2 \
    --online_eval_every_n_env_steps=2000 \
    --eval_n_trajs=10 \
    --n_train_step_per_epoch_offline=1000 \
    --n_pretrain_epochs=1000 \
    --max_online_env_steps=$max_online_env_steps \
    --mixing_ratio=0.5 \
    --reward_scale=1.0 \
    --reward_bias=0.0 \
    --enable_calql=True \
    --sarsa_lb=True \
    --logging.output_dir=/tmp/Cal_QL_gym_${env}_${now} &"
    
    echo -e "$command\n"

    if [[ $exp_num -lt $num_max ]]; then
        eval $command
    fi

    sleep 20
    exp_num=$((exp_num+1))
done
done
done