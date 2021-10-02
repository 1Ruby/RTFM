# Training on GridWorld

The environment grid world with four rooms is available.

![fourrooms_gate](https://tva1.sinaimg.cn/large/008i3skNly1gv14h0v5gkj602w02w0mv02.jpg)

## Basic commands:

Training on FourroomsGate env

```
python run_exp.py --env GridWorld:fourrooms_gate_rtfm-v0 --model paper_txt2pi --demb 30 --drnn_small 10 --drnn 100 --drep 400 --num_actors 20 --test_actors 3 --batch_size 24 --total_frames=100000000  --learning_rate 0.0001 --entropy_cost 0.01 --xpid fourrooms_gate_rtfm_0
```

Using tensorboard

```
tensorboard --samples_per_plugin images=1e9 --logdir runs/fourrooms_gate_rtfm_0
```

Make statistic image

```
python make_img.py --logs_dir checkpoints/fourrooms_gate_rtfm_0 --save_dir images/fourrooms_gate_rtfm_0 --fig_id 0 --subtitle 'Basic test'
```

