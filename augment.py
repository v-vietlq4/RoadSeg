import imgaug.augmenters as iaa

aug_transform = iaa.SomeOf((0, None), [
    iaa.OneOf([
        iaa.MultiplyAndAddToBrightness(mul=(0.3, 1.6), add=(-50, 50)),
        iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
        iaa.ChannelShuffle(0.5),
        iaa.RemoveSaturation(),
        iaa.Grayscale(alpha=(0.0, 1.0)),
        iaa.ChangeColorTemperature((1100, 35000)),
    ]),
    iaa.OneOf([
                iaa.MedianBlur(k=(3, 7)),
                iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
                iaa.MotionBlur(k=(3, 9), angle=[-45, 45]),
                iaa.MeanShiftBlur(spatial_radius=(5.0, 10.0), color_radius=(5.0, 10.0)),
                iaa.AllChannelsCLAHE(clip_limit=(1, 10)),
                iaa.AllChannelsHistogramEqualization(),
                iaa.GammaContrast((0.5, 1.5), per_channel=True),
                iaa.GammaContrast((0.5, 1.5)),
                iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
                iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
                iaa.HistogramEqualization(),
                iaa.Sharpen(alpha=0.5)
            ]),
    iaa.OneOf([
        iaa.AveragePooling([2, 3]),
        iaa.MaxPooling(([2,3],[2,3])),                
    ]),
    iaa.OneOf([
        iaa.Clouds(),
        iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
        iaa.Rain(speed=(0.1, 0.3))
    ])
        ], random_order=True)
def get_color_augmentation(augment_prob):
    return iaa.Sometimes(augment_prob, aug_transform).augment_image