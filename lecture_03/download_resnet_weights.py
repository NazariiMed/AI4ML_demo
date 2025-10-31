"""
Helper script to manually download ResNet-18 pre-trained weights.

This is useful if you:
1. Have network issues during demo
2. Want to prepare offline demo materials
3. Need to verify the download works before the lecture

The weights will be cached at: ~/.cache/torch/hub/checkpoints/
"""

import torch
import torchvision.models as models
from pathlib import Path
import logging
import ssl

# Disable SSL verification (for corporate/university networks)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def download_resnet_weights():
    """Download ResNet-18 pre-trained weights."""
    logger.info("="*70)
    logger.info("DOWNLOADING RESNET-18 PRE-TRAINED WEIGHTS")
    logger.info("="*70)
    
    cache_dir = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'
    logger.info(f"\nWeights will be cached at:")
    logger.info(f"  {cache_dir}")
    
    weight_file = cache_dir / 'resnet18-f37072fd.pth'
    
    if weight_file.exists():
        logger.info(f"\n✓ Weights already downloaded!")
        logger.info(f"  File size: {weight_file.stat().st_size / 1024 / 1024:.1f} MB")
        return True
    
    logger.info(f"\nAttempting to download pre-trained weights...")
    logger.info(f"Source: https://download.pytorch.org/models/resnet18-f37072fd.pth")
    logger.info(f"Size: ~45 MB")
    
    try:
        # This will trigger the download
        model = models.resnet18(pretrained=True)
        logger.info(f"\n✓ Successfully downloaded ResNet-18 weights!")
        logger.info(f"  Saved to: {weight_file}")
        logger.info(f"  File size: {weight_file.stat().st_size / 1024 / 1024:.1f} MB")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Download failed: {e}")
        logger.info(f"\nTroubleshooting:")
        logger.info(f"  1. Check internet connection")
        logger.info(f"  2. Try using a VPN if behind a firewall")
        logger.info(f"  3. Manually download from:")
        logger.info(f"     https://download.pytorch.org/models/resnet18-f37072fd.pth")
        logger.info(f"  4. Save to: {cache_dir}")
        return False

if __name__ == '__main__':
    success = download_resnet_weights()
    
    if success:
        logger.info("\n" + "="*70)
        logger.info("READY FOR DEMO!")
        logger.info("="*70)
        logger.info("You can now run 02_cnn_defect_detection.py offline.")
    else:
        logger.info("\n" + "="*70)
        logger.info("FALLBACK OPTION")
        logger.info("="*70)
        logger.info("The demo will automatically train from scratch if weights")
        logger.info("are not available. This will take longer but still works!")
