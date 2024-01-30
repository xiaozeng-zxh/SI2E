# A2C
for task in 'DoorKey-8x8' 'Empty-16x16'
do
    for seed in 1 10
    do
        python3 -m scripts.train_cuda_0 --algo a2c --env MiniGrid-$task-v0 --model $task/MiniGrid-$task-v0-original-$seed \
        --save-interval 100 --frames 3000000 --seed $seed --use_batch
    done
done