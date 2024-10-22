# A2C + SE
for task in 'DoorKey-8x8'
do
    for seed in 1
    do
        python3 -m scripts.train_cuda_0 --algo a2c --env MiniGrid-$task-v0 --model $task/MiniGrid-$task-v0-sent-$seed \
        --save-interval 100 --frames 3000000 --use_entropy_reward --seed $seed --beta 0.005
    done
done