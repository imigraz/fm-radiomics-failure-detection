"""Foundation-model embeddings (DINOv3, RadDINO, 3DINO) in the four paper modes."""

# Paper modes: what the FM sees, and how its tokens are pooled.
#   img  CLS token of the plain image
#   ov   CLS token of the image with the predicted mask as a colour overlay
#   w    patch tokens of the plain image, pooled with mask-coverage weights
#   ovw  patch tokens of the overlay image, pooled with mask-coverage weights
MODES = ("img", "ov", "w", "ovw")
BACKBONES = ("dinov3", "raddino", "3dino")
