---
models/ - NEURAL NETWORK ARCHITECTURES
---

PURPOSE:
Contains implementations of various neural network architectures for CIFAR-100
and ImageNet, including ResNets, VGGs, MobileNets, and ShuffleNets.

---
FILE: models/__init__.py
---

PURPOSE: Model registry that maps model names to their constructors

STRUCTURE:
  model_dict = {
      'resnet38': resnet38,
      'resnet110': resnet110,
      'resnet8x4': resnet8x4,
      'resnet32x4': resnet32x4,
      'vgg8': vgg8_bn,
      'MobileNetV2': mobile_half,
      'ShuffleV2': ShuffleV2,
      'ResNet50': resnet50,
      ...
  }

USAGE:
  from models import model_dict

  model = model_dict['resnet32x4'](num_classes=100)
  # Creates ResNet32x4 for CIFAR-100

NAMING CONVENTION:
  resnetXXxY:
  - XX: Number of layers
  - Y: Width multiplier

  Examples:
  - resnet8x4: 8 layers, 4× width = [32, 64, 128, 256] channels
  - resnet32x4: 32 layers, 4× width = [32, 64, 128, 256] channels
  - resnet110: 110 layers, 1× width = [16, 32, 64] channels

AVAILABLE MODELS:

CIFAR-100 Models:
- ResNets: resnet38, resnet110, resnet116, resnet8x4, resnet14x4, resnet32x4, resnet38x4
- VGGs: vgg8_bn, vgg13_bn
- MobileNets: mobile_half, mobile_half_double
- ShuffleNets: ShuffleV1, ShuffleV2, ShuffleV2_1_5

ImageNet Models:
- ResNets: resnet18, resnet34, resnet50, resnext50_32x4d
- Wide ResNets: wide_resnet10_2, wide_resnet18_2, wide_resnet34_2, wide_resnet50_2
- MobileNets: mobilenet_v2
- ShuffleNets: shufflenet_v2_x1_0

---
FILE: models/resnet.py (CIFAR-100)
---

PURPOSE: ResNet implementation for CIFAR-100 (32×32 images)

KEY CLASSES:
1. BasicBlock: Basic residual block (3×3 → 3×3)
2. Bottleneck: Bottleneck block (1×1 → 3×3 → 1×1)
3. ResNet: Main ResNet class

BASICBLOCK:
  class BasicBlock(nn.Module):
      expansion = 1

      def __init__(self, inplanes, planes, stride=1, downsample=None):
          super(BasicBlock, self).__init__()
          self.conv1 = conv3x3(inplanes, planes, stride)
          self.bn1 = nn.BatchNorm2d(planes)
          self.relu = nn.ReLU(inplace=True)
          self.conv2 = conv3x3(planes, planes)
          self.bn2 = nn.BatchNorm2d(planes)
          self.downsample = downsample
          self.stride = stride

      def forward(self, x):
          residual = x

          out = self.conv1(x)
          out = self.bn1(out)
          out = self.relu(out)

          out = self.conv2(out)
          out = self.bn2(out)

          if self.downsample is not None:
              residual = self.downsample(x)

          out += residual  # Skip connection!
          out = self.relu(out)
          return out

  FLOW:
    Input x → conv1 → bn1 → relu → conv2 → bn2 → (+residual) → relu → Output

  SKIP CONNECTION:
    out = conv_bn_relu_conv_bn(x) + x

  Why?
  - Allows gradients to flow directly backward
  - Enables training very deep networks
  - Residual learning: Learn F(x) = H(x) - x instead of H(x)

BOTTLENECK:
  class Bottleneck(nn.Module):
      expansion = 4

      def __init__(self, inplanes, planes, stride=1, downsample=None):
          super(Bottleneck, self).__init__()
          self.conv1 = conv1x1(inplanes, planes)
          self.bn1 = nn.BatchNorm2d(planes)
          self.conv2 = conv3x3(planes, planes, stride=1)
          self.bn2 = nn.BatchNorm2d(planes)
          self.conv3 = conv1x1(planes, planes * 4)
          self.bn3 = nn.BatchNorm2d(planes * 4)
          self.relu = nn.ReLU(inplace=True)
          self.downsample = downsample
          self.stride = stride

  FLOW:
    Input x → 1×1 conv (compress) → 3×3 conv → 1×1 conv (expand) → (+residual) → Output

  CHANNEL FLOW:
    [256] → [64] → [64] → [256]
    Bottleneck: Reduces then restores channels

  WHY BOTTLENECK?
  - Fewer parameters: 1×1 + 3×3 + 1×1 < 3×3 + 3×3
  - Enables deeper networks
  - Used in ResNet-50, ResNet-101, ResNet-152

RESNET CLASS:
  class ResNet(nn.Module):
      def __init__(self, depth, num_filters, block_name='BasicBlock', num_classes=10):
          super(ResNet, self).__init__()

          # Determine number of blocks per layer
          if block_name.lower() == 'basicblock':
              n = (depth - 2) // 6
              block = BasicBlock
          elif block_name.lower() == 'bottleneck':
              n = (depth - 2) // 9
              block = Bottleneck

          self.inplanes = num_filters[0]
          self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
          self.bn1 = nn.BatchNorm2d(num_filters[0])
          self.relu = nn.ReLU(inplace=True)
          self.layer1 = self._make_layer(block, num_filters[1], n)
          self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
          self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
          self.avgpool = nn.AdaptiveAvgPool2d((1,1))
          self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)

ARCHITECTURE EXAMPLE (resnet32x4):
  depth = 32, num_filters = [32, 32, 64, 128, 256]

  Calculation:
    n = (32 - 2) // 6 = 5 blocks per layer

  Input: [B, 3, 32, 32]
    ↓
  conv1: [B, 32, 32, 32]  # num_filters[0] = 32
  bn1, relu
    ↓
  layer1 (5 blocks): [B, 32, 32, 32]  # num_filters[1] = 32, no downsampling
    ↓
  layer2 (5 blocks): [B, 64, 16, 16]  # num_filters[2] = 64, stride=2
    ↓
  layer3 (5 blocks): [B, 128, 8, 8]  # num_filters[3] = 128, stride=2
    ↓
  avgpool: [B, 128, 1, 1] → [B, 128]
    ↓
  fc: [B, 100]

  Total layers: 1 + 5×2 + 5×2 + 5×2 + 1 = 32 ✓

FORWARD METHOD (KEY FOR DISTILLATION):
  def forward(self, x, is_feat=False):
      x = self.conv1(x)
      x = self.bn1(x)
      f0 = self.relu(x)

      f1 = self.layer1(f0)
      f2 = self.layer2(f1)
      f3 = self.layer3(f2)

      f4 = self.avgpool(f3)
      f4 = f4.view(f4.size(0), -1)

      out = self.fc(f4)

      if is_feat:
          return [f0, f1, f2, f3, f4], out
      else:
          return out

  FEATURE EXTRACTION:
    When is_feat=True, returns intermediate features:
    - f0: After initial conv+bn+relu
    - f1: After layer1
    - f2: After layer2
    - f3: After layer3
    - f4: After avgpool (pooled features)
    - out: Final logits

  USAGE FOR DISTILLATION:
    feat_s, logit_s = model_s(images, is_feat=True)
    feat_t, logit_t = model_t(images, is_feat=True)

    # Can now transfer knowledge at multiple layers
    loss_hint = MSE(feat_s[2], feat_t[2])  # Layer 2
    loss_pkt = PKT(feat_s[4], feat_t[4])   # Final features

get_feat_modules():
  def get_feat_modules(self):
      feat_m = nn.ModuleList([])
      feat_m.append(self.conv1)
      feat_m.append(self.bn1)
      feat_m.append(self.relu)
      feat_m.append(self.layer1)
      feat_m.append(self.layer2)
      feat_m.append(self.layer3)
      feat_m.append(self.avgpool)
      feat_m.append(self.fc)
      return feat_m

  PURPOSE: Returns list of feature extraction modules

  USAGE:
    cls_t = model_t.get_feat_modules()[-1]  # Get classifier (fc layer)

    Used in SRRL and SimKD to access teacher's classifier

WEIGHT INITIALIZATION:
  for m in self.modules():
      if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)

  Kaiming (He) initialization:
  - Designed for ReLU activations
  - Prevents vanishing/exploding gradients
  - Variance preserved through layers

FACTORY FUNCTIONS:
  def resnet8x4(**kwargs):
      return ResNet(8, [32, 64, 128, 256], 'basicblock', **kwargs)

  def resnet32x4(**kwargs):
      return ResNet(32, [32, 64, 128, 256], 'basicblock', **kwargs)

  def resnet110(**kwargs):
      return ResNet(110, [16, 16, 32, 64], 'basicblock', **kwargs)

---
MODEL COMPARISON
---

RESNET VARIANTS FOR CIFAR-100:

| Model        | Depth | Width      | Params  | Top-1 Acc |
|--------------|-------|------------|---------|-----------|
| resnet110    | 110   | [16,32,64] | ~1.7M   | ~77%      |
| resnet8x4    | 8     | [32,64,128,256] | ~0.015M | ~69%      |
| resnet32x4   | 32    | [32,64,128,256] | ~2.8M   | ~78%      |
| resnet38x4   | 38    | [32,64,128,256] | ~3.2M   | ~79%      |

TYPICAL TEACHER-STUDENT PAIRS:
- Teacher: resnet32x4 (2.8M params)
  Student: resnet8x4 (0.015M params)
  Ratio: ~187x smaller

- Teacher: resnet110 (1.7M params)
  Student: resnet38 (0.5M params)
  Ratio: ~3.4x smaller

---
FILE: models/util.py
---

PURPOSE: Projection and transformation modules for knowledge distillation

CONVREGCONV:
  class ConvReg(nn.Module):
      """Convolutional regression for FitNet"""
      def __init__(self, s_shape, t_shape, use_relu=True):
          super(ConvReg, self).__init__()
          self.use_relu = use_relu
          s_N, s_C, s_H, s_W = s_shape
          t_N, t_C, t_H, t_W = t_shape
          if s_H == 2 * t_H:
              self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
          elif s_H * 2 == t_H:
              self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
          elif s_H >= t_H:
              self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
          else:
              raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
          self.bn = nn.BatchNorm2d(t_C)
          self.relu = nn.ReLU(inplace=True)

      def forward(self, x):
          x = self.conv(x)
          if self.use_relu:
              return self.relu(self.bn(x))
          else:
              return self.bn(x)

  PURPOSE: Aligns student features to match teacher feature dimensions

  CASES:
  1. Student 2× larger spatial size:
     student: [B, 128, 32, 32]
     teacher: [B, 256, 16, 16]
     → Use stride=2 conv to downsample

  2. Student 2× smaller spatial size:
     student: [B, 128, 8, 8]
     teacher: [B, 256, 16, 16]
     → Use transpose conv to upsample

  3. Other size differences:
     → Use adaptive kernel size

  USAGE IN FITNET:
    regress_s = ConvReg(feat_s[2].shape, feat_t[2].shape)
    f_s_aligned = regress_s(feat_s[2])
    loss = MSE(f_s_aligned, feat_t[2])

SELFA:
  class SelfA(nn.Module):
      """Self-attention for SemCKD"""

      def __init__(self, batch_size, s_n, t_n, soft_alignment):
          ...

  PURPOSE: Learns cross-layer attention between student and teacher

  CONCEPT:
    Which teacher layer is most useful for each student layer?

    Student layers: [64, 128, 256]
    Teacher layers: [128, 256, 512]

    Attention matrix:
               T_layer1  T_layer2  T_layer3
    S_layer1 [  0.6       0.3       0.1   ]
    S_layer2 [  0.2       0.7       0.1   ]
    S_layer3 [  0.1       0.3       0.6   ]

    Student layer 1 learns mostly from teacher layer 1
    Student layer 2 learns mostly from teacher layer 2
    etc.

SRRL:
  class SRRL(nn.Module):
      """Student Representation Refinement Learning"""

      def __init__(self, s_n, t_n):
          super(SRRL, self).__init__()
          self.t_n = t_n
          self.s_n = s_n

          # Transform student features
          self.transform = nn.Sequential(
              nn.Conv2d(s_n, t_n, kernel_size=1),
              nn.BatchNorm2d(t_n),
              nn.ReLU(inplace=True)
          )

      def forward(self, feat_s, cls_t):
          # Transform student features to teacher dimension
          trans_feat_s = self.transform(feat_s)
          trans_feat_s_flat = F.adaptive_avg_pool2d(trans_feat_s, (1, 1))
          trans_feat_s_flat = trans_feat_s_flat.view(trans_feat_s_flat.size(0), -1)

          # Predict using teacher's classifier
          pred_feat_s = cls_t(trans_feat_s_flat)

          return trans_feat_s, pred_feat_s

  PURPOSE: Transform student features, then use teacher's classifier

SIMKD:
  class SimKD(nn.Module):
      """Simple Knowledge Distillation"""

      def __init__(self, s_n, t_n, factor=2):
          super(SimKD, self).__init__()
          self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

          # Bottleneck design
          def conv1x1(in_channels, out_channels, stride=1):
              return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               padding=0, stride=stride, bias=False)
          def conv3x3(in_channels, out_channels, stride=1):
              return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, stride=stride, bias=False)

          # Channel alignment
          self.group1 = nn.Sequential(
              conv1x1(s_n, t_n // factor),
              nn.BatchNorm2d(t_n // factor),
              nn.ReLU(inplace=True),
              conv3x3(t_n // factor, t_n // factor),
              nn.BatchNorm2d(t_n // factor),
              nn.ReLU(inplace=True),
              conv1x1(t_n // factor, t_n),
              nn.BatchNorm2d(t_n),
              nn.ReLU(inplace=True),
          )

      def forward(self, feat_s, feat_t, cls_t):
          # Align student features
          trans_feat_s = self.group1(feat_s)
          trans_feat_t = feat_t

          # Spatial alignment
          trans_feat_s = self.avg_pool(trans_feat_s)
          trans_feat_t = self.avg_pool(trans_feat_t)

          # Flatten
          trans_feat_s = trans_feat_s.view(trans_feat_s.size(0), -1)
          trans_feat_t = trans_feat_t.view(trans_feat_t.size(0), -1)

          # Use teacher's classifier for prediction
          pred_feat_s = cls_t(trans_feat_s)

          return trans_feat_s, trans_feat_t, pred_feat_s

  BOTTLENECK STRUCTURE:
    [256] → [128] → [128] → [512]
    (s_n)   (t_n/2) (t_n/2)  (t_n)

    Reduces channels first, then expands
    Fewer parameters than direct 1×1 conv

  KEY INNOVATION:
    Student doesn't have its own classifier!
    Uses teacher's well-calibrated classifier instead.

---
TYPICAL MODEL USAGE
---

CREATE MODEL:
  from models import model_dict

  # For CIFAR-100
  model = model_dict['resnet32x4'](num_classes=100)

  # For ImageNet
  model = model_dict['ResNet50'](num_classes=1000)

STANDARD FORWARD:
  output = model(images)
  # output: [B, num_classes]

FEATURE EXTRACTION:
  features, logits = model(images, is_feat=True)
  # features: List of [f0, f1, f2, f3, f4]
  # logits: [B, num_classes]

LOAD PRE-TRAINED:
  checkpoint = torch.load('resnet32x4_best.pth')
  model.load_state_dict(checkpoint['model'])

COUNT PARAMETERS:
  num_params = sum(p.numel() for p in model.parameters())
  print(f"Parameters: {num_params / 1e6:.2f}M")

---
END OF models/ EXPLANATION
---
