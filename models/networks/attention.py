class BaselineSpatialAttention(nn.Module):
    def __init__(self, opt):
        super(BaselineSpatialAttention, self).__init__()

        # create vgg Model
        self.vgg_feature_extracter = VGGFeatureExtractor(opt)

        self.non_local_blk = NonLocalBlock(256)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, tgt, ref):
        tgt_feature = self.vgg_feature_extracter(tgt)
        ref_feature = self.vgg_feature_extracter(ref)

        ref_value = self.vgg_feature_extracter(ref, isValue=True)

        attention, conf_map, out = self.non_local_blk(ref_feature, tgt_feature, ref_value)

        return attention, conf_map, out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)