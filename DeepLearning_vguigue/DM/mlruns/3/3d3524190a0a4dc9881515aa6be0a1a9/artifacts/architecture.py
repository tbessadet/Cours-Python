SpectralLTAE(
  (enc): CanalEncoder(
    (net): Sequential(
      (0): Sequential(
        (0): Conv2d(13, 416, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=13, bias=False)
        (1): BatchNorm2d(416, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (2): Sequential(
        (0): Conv2d(416, 832, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=13, bias=False)
        (1): BatchNorm2d(832, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (4): Sequential(
        (0): Conv2d(832, 1664, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=13, bias=False)
        (1): BatchNorm2d(1664, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (5): AdaptiveAvgPool2d(output_size=1)
      (6): Flatten(start_dim=1, end_dim=-1)
    )
  )
  (ltae): LTAE(
    (norm_in): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (fc_k): Linear(in_features=128, out_features=128, bias=True)
    (drop): Dropout(p=0.2, inplace=False)
    (norm_out): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  )
  (head): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Dropout(p=0.3, inplace=False)
    (4): Linear(in_features=128, out_features=10, bias=True)
  )
)