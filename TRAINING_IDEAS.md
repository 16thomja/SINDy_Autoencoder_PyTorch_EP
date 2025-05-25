[] x.clamp(0.0, 1.0)
[] nn.Hardtanh(0, 1) as final activation
[] Multiply pixel-wise MSE by (alpha * pixel brightness), where alpha is tunable
[] L1 or Huber loss (MSE is comparatively lenient with small deviations)
