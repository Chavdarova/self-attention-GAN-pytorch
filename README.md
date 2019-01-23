# self-attention-GAN-pytorch


In this fork of the repo I make some modifications to support (unconditional GAN) training on CelebA, for 64px images. This is in the `unconditional_64px` branch. My running script looks like:

```
python train.py --data_path /Tmp/beckhamc/img_align_celeba --dataset celeba --batch_size 32 --imsize 64 --name sagan --model_save_step 2000 --g_lr 2e-4 --d_lr 2e-4 --d_steps_per_iter 5 --num_workers 4
```

39620 iters:

![image](https://user-images.githubusercontent.com/2417792/51616874-0d47a580-1ef9-11e9-9d78-f005c0698b41.png)


@article{Zhang2018SelfAttentionGA,
    title={Self-Attention Generative Adversarial Networks},
    author={Han Zhang and Ian J. Goodfellow and Dimitris N. Metaxas and Augustus Odena},
    journal={CoRR},
    year={2018},
    volume={abs/1805.08318}
}
