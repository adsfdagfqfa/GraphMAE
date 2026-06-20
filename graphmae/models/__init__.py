from .edcoder import PreModel
from .edcoder_degree_mask import DegreeMixedMaskPreModel


def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_rate = args.mask_rate
    drop_edge_rate = args.drop_edge_rate
    replace_rate = args.replace_rate
    mask_strategy = getattr(args, "mask_strategy", "random")
    degree_mask_ratio = getattr(args, "degree_mask_ratio", 0.3)
    degree_mask_power = getattr(args, "degree_mask_power", 1.0)


    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l
    concat_hidden = args.concat_hidden
    num_features = args.num_features

    model_cls = PreModel
    model_kwargs = {}
    if mask_strategy == "degree_mixed":
        model_cls = DegreeMixedMaskPreModel
        model_kwargs.update(
            degree_mask_ratio=degree_mask_ratio,
            degree_mask_power=degree_mask_power,
        )
    elif mask_strategy != "random":
        raise NotImplementedError(f"{mask_strategy} is not implemented.")

    model = model_cls(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
        **model_kwargs,
    )
    return model
