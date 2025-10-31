def make_blocks(
        block_fns: Tuple[Union[Type[BasicBlock], Type[Bottleneck]], ...],
        channels: Tuple[int, ...],
        block_repeats: Tuple[int, ...],
        inplanes: int,
        reduce_first: int = 1,
        output_stride: int = 32,
        down_kernel_size: int = 1,
        **kwargs,
) -> List[Tuple[str, nn.Module]]:
    """Create ResNet stages with specified block configurations.

    Args:
        block_fns: Block class to use for each stage.
        channels: Number of channels for each stage.
        block_repeats: Number of blocks to repeat for each stage.
        inplanes: Number of input channels.
        reduce_first: Reduction factor for first convolution in each stage.
        output_stride: Target output stride of network.
        down_kernel_size: Kernel size for downsample layers.

    Returns:
        Tuple of stage modules list and feature info list.
    """
    channels = [64, 128, 256, 512] if channels is None else channels

    stages = []
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (block_fn, planes, num_blocks) in enumerate(zip(
            block_fns,
            channels,
            block_repeats)
    ):
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride
        downsample_stage = stride != 1

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
            )
            downsample = downsample_avg(**down_kwargs)

        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            blocks.append(block_fn(
                inplanes,
                planes,
                stride,
                downsample,
                first_dilation=prev_dilation,
                reduce_first=reduce_first,
                dilation=dilation,
            ))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append(nn.Sequential(*blocks))

    return stages


class BasicBlock(nn.Module):
    """Basic residual block for ResNet.

    This is the standard residual block used in ResNet-18 and ResNet-34.
    """
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer class.
            norm_layer: Normalization layer class.
            attn_layer: Attention layer class.
            aa_layer: Anti-aliasing layer class.
            drop_block: DropBlock layer class.
            drop_path: Optional DropPath layer instance.
        """
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes,
            first_planes,
            kernel_size=3,
            stride=1 if use_aa else stride,
            padding=first_dilation,
            dilation=first_dilation,
            bias=False
        )
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet.

    This is the bottleneck block used in ResNet-50, ResNet-101, and ResNet-152.
    """
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            outplanes: int,
            pinch: int = 4,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer class.
            norm_layer: Normalization layer class.
            attn_layer: Attention layer class.
            aa_layer: Anti-aliasing layer class.
            drop_block: DropBlock layer class.
            drop_path: Optional DropPath layer instance.
        """
        super(Bottleneck, self).__init__()

        planes = outplanes // pinch

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes,
            width,
            kernel_size=3,
            stride=1 if use_aa else stride,
            padding=first_dilation,
            dilation=first_dilation,
            groups=cardinality,
            bias=False
        )
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path
