import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ implementation for semantic segmentation
    """
    def __init__(self, num_classes=1, backbone='resnet50', pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            self.backbone_model = models.resnet50(pretrained=pretrained)
            self.backbone_layers = nn.ModuleDict({
                'layer1': self.backbone_model.layer1,
                'layer2': self.backbone_model.layer2,
                'layer3': self.backbone_model.layer3,
                'layer4': self.backbone_model.layer4,
            })
            self.backbone_conv1 = self.backbone_model.conv1
            self.backbone_bn1 = self.backbone_model.bn1
            self.backbone_relu = self.backbone_model.relu
            self.backbone_maxpool = self.backbone_model.maxpool
            
            # ASPP module
            self.aspp = ASPP(2048, 256)
            
            # Decoder
            self.decoder = Decoder(256, 256, num_classes)
            
        elif backbone == 'resnet101':
            self.backbone_model = models.resnet101(pretrained=pretrained)
            self.backbone_layers = nn.ModuleDict({
                'layer1': self.backbone_model.layer1,
                'layer2': self.backbone_model.layer2,
                'layer3': self.backbone_model.layer3,
                'layer4': self.backbone_model.layer4,
            })
            self.backbone_conv1 = self.backbone_model.conv1
            self.backbone_bn1 = self.backbone_model.bn1
            self.backbone_relu = self.backbone_model.relu
            self.backbone_maxpool = self.backbone_model.maxpool
            
            # ASPP module
            self.aspp = ASPP(2048, 256)
            
            # Decoder
            self.decoder = Decoder(256, 256, num_classes)
    
    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Backbone forward pass
        x = self.backbone_conv1(x)
        x = self.backbone_bn1(x)
        x = self.backbone_relu(x)
        x = self.backbone_maxpool(x)
        
        x = self.backbone_layers['layer1'](x)
        low_level_features = x
        
        x = self.backbone_layers['layer2'](x)
        x = self.backbone_layers['layer3'](x)
        x = self.backbone_layers['layer4'](x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        x = self.decoder(x, low_level_features)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        
        self.conv6 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))
        
        x5 = self.global_avg_pool(x)
        x5 = self.relu(self.bn5(self.conv5(x5)))
        x5 = F.interpolate(x5, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.relu(self.bn6(self.conv6(x)))
        
        return x

class Decoder(nn.Module):
    """
    Decoder module for DeepLabV3+
    """
    def __init__(self, in_channels, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels + 48, 256, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, x, low_level_features):
        low_level_features = self.relu(self.bn1(self.conv1(low_level_features)))
        
        x = F.interpolate(x, size=low_level_features.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_level_features], dim=1)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        
        return x

class SegNet(nn.Module):
    """
    SegNet implementation for semantic segmentation
    """
    def __init__(self, num_classes=1, in_channels=3):
        super(SegNet, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Encoder
        self.encoder_conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.encoder_bn1 = nn.BatchNorm2d(64)
        self.encoder_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.encoder_bn2 = nn.BatchNorm2d(64)
        
        self.encoder_conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.encoder_bn3 = nn.BatchNorm2d(128)
        self.encoder_conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.encoder_bn4 = nn.BatchNorm2d(128)
        
        self.encoder_conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.encoder_bn5 = nn.BatchNorm2d(256)
        self.encoder_conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.encoder_bn6 = nn.BatchNorm2d(256)
        self.encoder_conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.encoder_bn7 = nn.BatchNorm2d(256)
        
        self.encoder_conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.encoder_bn8 = nn.BatchNorm2d(512)
        self.encoder_conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.encoder_bn9 = nn.BatchNorm2d(512)
        self.encoder_conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.encoder_bn10 = nn.BatchNorm2d(512)
        
        self.encoder_conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.encoder_bn11 = nn.BatchNorm2d(512)
        self.encoder_conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.encoder_bn12 = nn.BatchNorm2d(512)
        self.encoder_conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.encoder_bn13 = nn.BatchNorm2d(512)
        
        # Decoder
        self.decoder_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.decoder_bn1 = nn.BatchNorm2d(512)
        self.decoder_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(512)
        self.decoder_conv3 = nn.Conv2d(512, 512, 3, padding=1)
        self.decoder_bn3 = nn.BatchNorm2d(512)
        
        self.decoder_conv4 = nn.Conv2d(512, 512, 3, padding=1)
        self.decoder_bn4 = nn.BatchNorm2d(512)
        self.decoder_conv5 = nn.Conv2d(512, 512, 3, padding=1)
        self.decoder_bn5 = nn.BatchNorm2d(512)
        self.decoder_conv6 = nn.Conv2d(512, 256, 3, padding=1)
        self.decoder_bn6 = nn.BatchNorm2d(256)
        
        self.decoder_conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.decoder_bn7 = nn.BatchNorm2d(256)
        self.decoder_conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.decoder_bn8 = nn.BatchNorm2d(256)
        self.decoder_conv9 = nn.Conv2d(256, 128, 3, padding=1)
        self.decoder_bn9 = nn.BatchNorm2d(128)
        
        self.decoder_conv10 = nn.Conv2d(128, 128, 3, padding=1)
        self.decoder_bn10 = nn.BatchNorm2d(128)
        self.decoder_conv11 = nn.Conv2d(128, 64, 3, padding=1)
        self.decoder_bn11 = nn.BatchNorm2d(64)
        
        self.decoder_conv12 = nn.Conv2d(64, 64, 3, padding=1)
        self.decoder_bn12 = nn.BatchNorm2d(64)
        self.decoder_conv13 = nn.Conv2d(64, num_classes, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Encoder
        x = self.relu(self.encoder_bn1(self.encoder_conv1(x)))
        x = self.relu(self.encoder_bn2(self.encoder_conv2(x)))
        x, indices1 = F.max_pool2d(x, 2, 2, return_indices=True)
        
        x = self.relu(self.encoder_bn3(self.encoder_conv3(x)))
        x = self.relu(self.encoder_bn4(self.encoder_conv4(x)))
        x, indices2 = F.max_pool2d(x, 2, 2, return_indices=True)
        
        x = self.relu(self.encoder_bn5(self.encoder_conv5(x)))
        x = self.relu(self.encoder_bn6(self.encoder_conv6(x)))
        x = self.relu(self.encoder_bn7(self.encoder_conv7(x)))
        x, indices3 = F.max_pool2d(x, 2, 2, return_indices=True)
        
        x = self.relu(self.encoder_bn8(self.encoder_conv8(x)))
        x = self.relu(self.encoder_bn9(self.encoder_conv9(x)))
        x = self.relu(self.encoder_bn10(self.encoder_conv10(x)))
        x, indices4 = F.max_pool2d(x, 2, 2, return_indices=True)
        
        x = self.relu(self.encoder_bn11(self.encoder_conv11(x)))
        x = self.relu(self.encoder_bn12(self.encoder_conv12(x)))
        x = self.relu(self.encoder_bn13(self.encoder_conv13(x)))
        x, indices5 = F.max_pool2d(x, 2, 2, return_indices=True)
        
        # Decoder
        x = F.max_unpool2d(x, indices5, 2, 2)
        x = self.relu(self.decoder_bn1(self.decoder_conv1(x)))
        x = self.relu(self.decoder_bn2(self.decoder_conv2(x)))
        x = self.relu(self.decoder_bn3(self.decoder_conv3(x)))
        
        x = F.max_unpool2d(x, indices4, 2, 2)
        x = self.relu(self.decoder_bn4(self.decoder_conv4(x)))
        x = self.relu(self.decoder_bn5(self.decoder_conv5(x)))
        x = self.relu(self.decoder_bn6(self.decoder_conv6(x)))
        
        x = F.max_unpool2d(x, indices3, 2, 2)
        x = self.relu(self.decoder_bn7(self.decoder_conv7(x)))
        x = self.relu(self.decoder_bn8(self.decoder_conv8(x)))
        x = self.relu(self.decoder_bn9(self.decoder_conv9(x)))
        
        x = F.max_unpool2d(x, indices2, 2, 2)
        x = self.relu(self.decoder_bn10(self.decoder_conv10(x)))
        x = self.relu(self.decoder_bn11(self.decoder_conv11(x)))
        
        x = F.max_unpool2d(x, indices1, 2, 2)
        x = self.relu(self.decoder_bn12(self.decoder_conv12(x)))
        x = self.decoder_conv13(x)
        
        return x

class HybridOilSpillModel(nn.Module):
    """
    Hybrid model combining DeepLabV3+ and SegNet for oil spill detection
    """
    def __init__(self, num_classes=1, backbone='resnet50', pretrained=True):
        super(HybridOilSpillModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Initialize both models
        self.deeplabv3_plus = DeepLabV3Plus(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
        self.segnet = SegNet(num_classes=num_classes)
        
        # Fusion layer
        self.fusion_conv = nn.Conv2d(num_classes * 2, num_classes, 1)
        self.fusion_bn = nn.BatchNorm2d(num_classes)
        self.fusion_relu = nn.ReLU(inplace=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(num_classes * 2, num_classes, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Get predictions from both models
        deeplabv3_output = self.deeplabv3_plus(x)
        segnet_output = self.segnet(x)
        
        # Concatenate outputs
        combined = torch.cat([deeplabv3_output, segnet_output], dim=1)
        
        # Apply attention mechanism
        attention_weights = self.attention(combined)
        
        # Weighted fusion
        weighted_combined = combined * attention_weights.repeat(1, 2, 1, 1)
        
        # Final fusion
        output = self.fusion_relu(self.fusion_bn(self.fusion_conv(weighted_combined)))
        
        return output

def create_model(model_type='hybrid', num_classes=1, backbone='resnet50', pretrained=True):
    """
    Create model based on specified type
    """
    if model_type == 'hybrid':
        return HybridOilSpillModel(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    elif model_type == 'deeplabv3_plus':
        return DeepLabV3Plus(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    elif model_type == 'segnet':
        return SegNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test model creation
    model = create_model('hybrid', num_classes=1)
    
    # Test forward pass
    x = torch.randn(1, 3, 512, 512)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
