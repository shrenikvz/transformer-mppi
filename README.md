# Transformer based Model Predictive Path Integral Control
This project presents a novel approach to improve the Model Predictive Path Integral (MPPI) control by using a transformer to initialize the mean control sequence. Traditional MPPI methods often struggle with sample efficiency and computational costs due to suboptimal initial rollouts. We propose TransformerMPPI, which uses a transformer trained on historical control data to generate informed initial mean control sequences. TransformerMPPI combines the strengths of the attention mechanism in transformers and sampling-based control, leading to improved computational performance and sample efficiency. The ability of the transformer to capture long-horizon patterns in optimal control sequences allows TransformerMPPI to start from a more informed control sequence, reducing the number of samples required, and accelerating convergence to optimal control sequence.

![TransformerMPPI](./figures/transformer_mppi.png)



