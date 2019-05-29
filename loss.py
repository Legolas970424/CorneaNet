import keras.backend as K
from keras.losses import binary_crossentropy
from keras.losses import categorical_crossentropy

from metric import jaccard_score, f_score

SMOOTH = 1e-12

# =========================== crossentropy Losses ===========================
def bce_loss(gt, pr):
    return K.mean(binary_crossentropy(gt, pr))

def cce_loss(gt, pr, class_weights=1.):
    return K.mean(categorical_crossentropy(gt, pr) * class_weights)
    

# ============================== Jaccard Losses ==============================
def jaccard_loss(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    return 1 - jaccard_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image)

def bce_jaccard_loss(gt, pr, bce_weight=1., smooth=SMOOTH, per_image=True):
    bce = K.mean(binary_crossentropy(gt, pr))
    loss = bce_weight * bce + jaccard_loss(gt, pr, smooth=smooth, per_image=per_image)
    return loss

def cce_jaccard_loss(gt, pr, cce_weight=1., class_weights=1., smooth=SMOOTH, per_image=True):
    cce = categorical_crossentropy(gt, pr) * class_weights
    cce = K.mean(cce)
    return cce_weight * cce + jaccard_loss(gt, pr, smooth=smooth, class_weights=class_weights, per_image=per_image)


# ============================== Dice Losses ================================
def dice_loss(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    return 1 - f_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image, beta=1.)

def bce_dice_loss(gt, pr, bce_weight=1., smooth=SMOOTH, per_image=True):
    bce = K.mean(binary_crossentropy(gt, pr))
    loss = bce_weight * bce + dice_loss(gt, pr, smooth=smooth, per_image=per_image)
    return loss

def cce_dice_loss(gt, pr, cce_weight=1., class_weights=1., smooth=SMOOTH, per_image=True):
    cce = categorical_crossentropy(gt, pr) * class_weights
    cce = K.mean(cce)
    return cce_weight * cce + dice_loss(gt, pr, smooth=smooth, class_weights=class_weights, per_image=per_image)

