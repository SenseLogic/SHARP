#!/usr/bin/env python3

# -- IMPORTS

from __future__ import annotations;
from typing import Any;

def _patch_torchvision_functional_tensor() -> None:

    import sys;

    module_name = "torchvision.transforms.functional_tensor";

    if module_name in sys.modules:

        return;

    import torchvision.transforms._functional_tensor as functional_tensor;

    sys.modules[ module_name ] = functional_tensor;

_patch_torchvision_functional_tensor();

try:

    import argparse;
    from basicsr.archs.rrdbnet_arch import RRDBNet;
    import cv2;
    import glob;
    import math;
    import numpy as np;
    import os;
    from realesrgan import RealESRGANer;
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact;
    import sys;
    import torch;
    from tqdm import tqdm;

except ImportError as import_error:

    print( f"Missing dependency: {import_error}", file=sys.stderr );
    print( "Install with:", file=sys.stderr );
    print( "  run install_packages_cuda.bat, install_packages_rocm.bat, or install_packages_cpu.bat", file=sys.stderr );
    sys.exit( 1 );

# -- CONSTANTS

DEFAULT_MODEL_NAME = "remacri";
MODEL_SCALE = 4;
MODEL_NAME_SET = {
    "realanime",
    "realdigital",
    "realesrgan",
    "realesrnet",
    "remacri",
    "ultramix",
    "ultrasharp",
    };
MODEL_BLOCK_COUNT = {
    "realdigital": 6,
    };
DEFAULT_MODEL_BLOCK_COUNT = 23;
SRVGG_MODEL_NAME_SET = {
    "realanime",
    };
DEFAULT_SRVGG_CONV_COUNT = 16;
DEFAULT_MAX_RATIO = 4.0;
DEFAULT_AVIF_COMPRESSION = 85;
DEFAULT_JPEG_COMPRESSION = 85;
DEFAULT_WEBP_COMPRESSION = 85;
DEFAULT_TILE_SIZE = 400;
DEFAULT_ALPHA_COLOR = "FFFFFF";
DEFAULT_ALPHA_MODE = "lanczos";
DEFAULT_OUTPUT_IMAGE_FILE_PATH_TEMPLATE = "{d}{s}.png";
ALPHA_MODE_SET = { "lanczos", "realesrnet", "remove" };
SUPPORTED_IMAGE_EXTENSION_SET = {
    ".avif",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
    };

APPLICATION_FOLDER_PATH = os.path.dirname( os.path.abspath( __file__ ) ) + "/";
MODEL_FOLDER_PATH = APPLICATION_FOLDER_PATH + "MODEL/";

# -- TYPES

class SharpRealESRGANer( RealESRGANer ):

    def __init__(
        self,
        scale: int,
        model_path: str,
        model: torch.nn.Module,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        half: bool = False,
        device: torch.device | None = None,
        gpu_id: int | None = None
        ) -> None:

        self.scale = scale;
        self.tile_size = tile;
        self.tile_pad = tile_pad;
        self.pre_pad = pre_pad;
        self.mod_scale = None;
        self.half = half;

        if gpu_id is not None:

            self.device = (
                torch.device(
                    f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
                    )
                if device is None
                else device
                );

        else:

            self.device = (
                torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                    )
                if device is None
                else device
                );

        state_dict = get_state_dict_from_model_checkpoint( model_path );

        model.load_state_dict( state_dict, strict=True );
        model.eval();
        self.model = model.to( self.device );

        if self.half:

            self.model = self.model.half();

    def process(
        self
        ) -> None:

        progress_bar = tqdm( total=1, unit="tile", desc="sharp", leave=False );

        try:

            self.output = self.model( self.img );
            progress_bar.update( 1 );

        finally:

            progress_bar.close();

    def tile_process(
        self
        ) -> None:

        batch, channel, height, width = self.img.shape;
        output_height = height * self.scale;
        output_width = width * self.scale;
        output_shape = ( batch, channel, output_height, output_width );

        self.output = self.img.new_zeros( output_shape );
        tiles_x = math.ceil( width / self.tile_size );
        tiles_y = math.ceil( height / self.tile_size );
        tile_count = tiles_x * tiles_y;

        progress_bar = tqdm(
            total=tile_count,
            unit="tile",
            desc="sharp",
            leave=False
            );

        try:

            for tile_y_index in range( tiles_y ):

                for tile_x_index in range( tiles_x ):

                    offset_x = tile_x_index * self.tile_size;
                    offset_y = tile_y_index * self.tile_size;
                    input_start_x = offset_x;
                    input_end_x = min( offset_x + self.tile_size, width );
                    input_start_y = offset_y;
                    input_end_y = min( offset_y + self.tile_size, height );
                    input_start_x_pad = max( input_start_x - self.tile_pad, 0 );
                    input_end_x_pad = min( input_end_x + self.tile_pad, width );
                    input_start_y_pad = max( input_start_y - self.tile_pad, 0 );
                    input_end_y_pad = min( input_end_y + self.tile_pad, height );
                    input_tile_width = input_end_x - input_start_x;
                    input_tile_height = input_end_y - input_start_y;
                    input_tile = (
                        self.img[
                            :,
                            :,
                            input_start_y_pad:input_end_y_pad,
                            input_start_x_pad:input_end_x_pad
                            ]
                        );

                    with torch.no_grad():

                        output_tile = self.model( input_tile );

                    output_start_x = input_start_x * self.scale;
                    output_end_x = input_end_x * self.scale;
                    output_start_y = input_start_y * self.scale;
                    output_end_y = input_end_y * self.scale;
                    output_start_x_tile = ( input_start_x - input_start_x_pad ) * self.scale;
                    output_end_x_tile = output_start_x_tile + input_tile_width * self.scale;
                    output_start_y_tile = ( input_start_y - input_start_y_pad ) * self.scale;
                    output_end_y_tile = output_start_y_tile + input_tile_height * self.scale;

                    self.output[
                        :,
                        :,
                        output_start_y:output_end_y,
                        output_start_x:output_end_x
                        ] = (
                        output_tile[
                            :,
                            :,
                            output_start_y_tile:output_end_y_tile,
                            output_start_x_tile:output_end_x_tile
                            ]
                        );

                    progress_bar.update( 1 );

        finally:

            progress_bar.close();

# -- FUNCTIONS

def normalize_folder_path(
    folder_path: str
    ) -> str:

    folder_path = folder_path.replace( "\\", "/" );

    if not folder_path.endswith( "/" ):

        folder_path += "/";

    return folder_path;

# ~~

def normalize_file_path(
    file_path: str
    ) -> str:

    return file_path.replace( "\\", "/" );

# ~~

def get_input_relative_file_path(
    input_image_folder_path: str,
    input_image_file_path: str
    ) -> str:

    normalized_input_image_file_path = normalize_file_path( input_image_file_path );

    if normalized_input_image_file_path.startswith( input_image_folder_path ):

        return normalized_input_image_file_path[ len( input_image_folder_path ): ];

    raise ValueError(
        f"Input image is not inside input folder: {input_image_file_path}"
        );

# ~~

def get_output_image_file_folder_path(
    output_image_file_path: str
    ) -> str:

    normalized_output_image_file_path = normalize_file_path( output_image_file_path );
    last_slash_index = normalized_output_image_file_path.rfind( "/" );

    if last_slash_index < 0:

        return "";

    return normalized_output_image_file_path[ :last_slash_index + 1 ];

# ~~

def parse_arguments(
    ) -> argparse.Namespace:

    argument_parser = (
        argparse.ArgumentParser(
            description="GPU-accelerated AI image upscaler",
            )
        );

    argument_parser.add_argument(
        "input_image_folder_path",
        help="Input image folder path (ending with /)"
        );

    argument_parser.add_argument(
        "output_image_folder_path",
        help="Output image folder path (ending with /)"
        );

    argument_parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Input image file path inclusion filter (e.g. **/*.png)"
        );

    argument_parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Input image file path exclusion filter"
        );

    argument_parser.add_argument(
        "--template",
        default=DEFAULT_OUTPUT_IMAGE_FILE_PATH_TEMPLATE,
        help=(
            "Output image file path template "
            f"(default: {DEFAULT_OUTPUT_IMAGE_FILE_PATH_TEMPLATE})"
            )
        );

    argument_parser.add_argument(
        "--min-width",
        type=int,
        default=0,
        help="Minimum output width in pixels (0 = no limit)"
        );

    argument_parser.add_argument(
        "--min-height",
        type=int,
        default=0,
        help="Minimum output height in pixels (0 = no limit)"
        );

    argument_parser.add_argument(
        "--max-ratio",
        type=float,
        default=DEFAULT_MAX_RATIO,
        help=f"Maximum output scaling ratio (default: {DEFAULT_MAX_RATIO})"
        );

    argument_parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="Maximum output width in pixels (0 = no limit)"
        );

    argument_parser.add_argument(
        "--max-height",
        type=int,
        default=0,
        help="Maximum output height in pixels (0 = no limit)"
        );

    argument_parser.add_argument(
        "--max-upscaled-width",
        type=int,
        default=0,
        help=(
            "Maximum output width in pixels when upscaling "
            "(0 = no limit)"
            )
        );

    argument_parser.add_argument(
        "--max-upscaled-height",
        type=int,
        default=0,
        help=(
            "Maximum output height in pixels when upscaling "
            "(0 = no limit)"
            )
        );

    argument_parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile size if GPU runs out of memory (default: {DEFAULT_TILE_SIZE})"
        );

    argument_parser.add_argument(
        "--model",
        choices=sorted( MODEL_NAME_SET ),
        default=DEFAULT_MODEL_NAME,
        help=f"Upscaling model (default: {DEFAULT_MODEL_NAME})"
        );

    argument_parser.add_argument(
        "--compression",
        type=int,
        default=None,
        help="Set AVIF, JPEG, and WebP compression quality"
        );

    argument_parser.add_argument(
        "--avif-compression",
        type=int,
        default=None,
        help=f"AVIF compression quality (default: {DEFAULT_AVIF_COMPRESSION})"
        );

    argument_parser.add_argument(
        "--jpeg-compression",
        type=int,
        default=None,
        help=f"JPEG compression quality (default: {DEFAULT_JPEG_COMPRESSION})"
        );

    argument_parser.add_argument(
        "--webp-compression",
        type=int,
        default=None,
        help=f"WebP compression quality (default: {DEFAULT_WEBP_COMPRESSION})"
        );

    argument_parser.add_argument(
        "--alpha-mode",
        choices=sorted( ALPHA_MODE_SET ),
        default=DEFAULT_ALPHA_MODE,
        help=(
            "Alpha channel handling for AVIF/PNG/WebP inputs: "
            "lanczos = Lanczos upscale and preserve alpha; "
            "realesrnet = AI upscale alpha with RealESRNet; "
            "remove = composite onto --alpha-color and drop alpha "
            f"(default: {DEFAULT_ALPHA_MODE})"
            )
        );

    argument_parser.add_argument(
        "--alpha-color",
        default=DEFAULT_ALPHA_COLOR,
        help=(
            f"Background color for alpha compositing (RRGGBB hex), "
            f"used with --alpha-mode remove and JPEG output (default: {DEFAULT_ALPHA_COLOR})"
            )
        );

    argument_parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip upscaling if output is newer than input"
        );

    compute_backend_group = argument_parser.add_mutually_exclusive_group();

    compute_backend_group.add_argument(
        "--cpu",
        action="store_const",
        const="cpu",
        dest="compute_backend",
        help="Force CPU computation"
        );

    compute_backend_group.add_argument(
        "--cuda",
        action="store_const",
        const="cuda",
        dest="compute_backend",
        help="Force NVIDIA CUDA computation"
        );

    compute_backend_group.add_argument(
        "--rocm",
        action="store_const",
        const="rocm",
        dest="compute_backend",
        help="Force AMD ROCm computation"
        );

    argument_parser.set_defaults( compute_backend=None );

    return argument_parser.parse_args();

# ~~

def resolve_compression_settings(
    command_line_arguments: argparse.Namespace
    ) -> tuple[ int, int, int ]:

    avif_compression = command_line_arguments.avif_compression;
    jpeg_compression = command_line_arguments.jpeg_compression;
    webp_compression = command_line_arguments.webp_compression;

    if command_line_arguments.compression is not None:

        if avif_compression is None:

            avif_compression = command_line_arguments.compression;

        if jpeg_compression is None:

            jpeg_compression = command_line_arguments.compression;

        if webp_compression is None:

            webp_compression = command_line_arguments.compression;

    if avif_compression is None:

        avif_compression = DEFAULT_AVIF_COMPRESSION;

    if jpeg_compression is None:

        jpeg_compression = DEFAULT_JPEG_COMPRESSION;

    if webp_compression is None:

        webp_compression = DEFAULT_WEBP_COMPRESSION;

    return avif_compression, jpeg_compression, webp_compression;

# ~~

def validate_input_image_folder_path(
    input_image_folder_path: str
    ) -> None:

    folder_path = input_image_folder_path.rstrip( "/" );

    if not os.path.isdir( folder_path ):

        print(
            f"Input image folder not found: {input_image_folder_path}",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def validate_maximum_ratio(
    maximum_ratio: float
    ) -> None:

    if maximum_ratio <= 0 or maximum_ratio > MODEL_SCALE:

        print(
            f"Maximum ratio must be greater than 0 and at most {MODEL_SCALE}: {maximum_ratio}",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def parse_alpha_color(
    alpha_color_hex: str
    ) -> tuple[ int, int, int ]:

    normalized_alpha_color_hex = alpha_color_hex.lstrip( "#" ).upper();

    if len( normalized_alpha_color_hex ) != 6:

        print(
            f"Alpha color must be a 6-digit hex value (RRGGBB): {alpha_color_hex}",
            file=sys.stderr
            );
        sys.exit( 1 );

    try:

        red = int( normalized_alpha_color_hex[ 0:2 ], 16 );
        green = int( normalized_alpha_color_hex[ 2:4 ], 16 );
        blue = int( normalized_alpha_color_hex[ 4:6 ], 16 );

    except ValueError:

        print(
            f"Alpha color must be a 6-digit hex value (RRGGBB): {alpha_color_hex}",
            file=sys.stderr
            );
        sys.exit( 1 );

    return blue, green, red;

# ~~

def validate_output_image_file_name_template(
    output_image_file_name_template: str
    ) -> None:

    output_extension = os.path.splitext( output_image_file_name_template )[ 1 ];

    if not is_loadable_image_extension( output_extension ):

        print(
            "Output image file name template must produce a "
            ".avif, .jpg, .jpeg, .png, or .webp file.",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def validate_minimum_dimensions(
    minimum_width: int,
    minimum_height: int
    ) -> None:

    if minimum_width < 0:

        print(
            f"Minimum width must be 0 or greater: {minimum_width}",
            file=sys.stderr
            );
        sys.exit( 1 );

    if minimum_height < 0:

        print(
            f"Minimum height must be 0 or greater: {minimum_height}",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def validate_maximum_upscaled_dimensions(
    maximum_upscaled_width: int,
    maximum_upscaled_height: int
    ) -> None:

    if maximum_upscaled_width < 0:

        print(
            f"Maximum upscaled width must be 0 or greater: {maximum_upscaled_width}",
            file=sys.stderr
            );
        sys.exit( 1 );

    if maximum_upscaled_height < 0:

        print(
            f"Maximum upscaled height must be 0 or greater: {maximum_upscaled_height}",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def get_model_weights_file_path(
    model_name: str
    ) -> str:

    model_weights_file_path = MODEL_FOLDER_PATH + model_name + ".pth";

    if not os.path.isfile( model_weights_file_path ):

        print(
            f"Model weights not found: {model_weights_file_path}",
            file=sys.stderr
            );
        sys.exit( 1 );

    return model_weights_file_path;

# ~~

def is_old_arch_state_dict(
    state_dict: dict[ str, Any ]
    ) -> bool:

    return any( key.startswith( "model." ) for key in state_dict );

# ~~

def convert_old_arch_state_dict_to_rrdbnet_state_dict(
    old_arch_state_dict: dict[ str, Any ]
    ) -> dict[ str, Any ]:

    rrdbnet = (
        RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=MODEL_SCALE
            )
        );
    rrdbnet_state_dict = rrdbnet.state_dict();
    pretrained_state_dict: dict[ str, Any ] = {};

    for key, value in old_arch_state_dict.items():

        if key.startswith( "module." ):

            pretrained_state_dict[ key[ 7: ] ] = value;

        else:

            pretrained_state_dict[ key ] = value;

    remaining_key_set = set( rrdbnet_state_dict.keys() );

    for key, value in rrdbnet_state_dict.items():

        if (
            key in pretrained_state_dict
            and pretrained_state_dict[ key ].size() == value.size()
            ):

            rrdbnet_state_dict[ key ] = pretrained_state_dict[ key ];
            remaining_key_set.discard( key );

    rrdbnet_state_dict[ "conv_first.weight" ] = (
        pretrained_state_dict[ "model.0.weight" ]
        );
    rrdbnet_state_dict[ "conv_first.bias" ] = (
        pretrained_state_dict[ "model.0.bias" ]
        );

    for key in list( remaining_key_set ):

        if "rdb" not in key:

            continue;

        old_arch_key = (
            key.replace( "body.", "model.1.sub." )
            .replace( "rdb", "RDB" )
            );

        if key.endswith( ".weight" ):

            old_arch_key = old_arch_key.replace( ".weight", ".0.weight" );

        elif key.endswith( ".bias" ):

            old_arch_key = old_arch_key.replace( ".bias", ".0.bias" );

        rrdbnet_state_dict[ key ] = pretrained_state_dict[ old_arch_key ];
        remaining_key_set.discard( key );

    rrdbnet_state_dict[ "conv_body.weight" ] = (
        pretrained_state_dict[ "model.1.sub.23.weight" ]
        );
    rrdbnet_state_dict[ "conv_body.bias" ] = (
        pretrained_state_dict[ "model.1.sub.23.bias" ]
        );
    rrdbnet_state_dict[ "conv_up1.weight" ] = (
        pretrained_state_dict[ "model.3.weight" ]
        );
    rrdbnet_state_dict[ "conv_up1.bias" ] = (
        pretrained_state_dict[ "model.3.bias" ]
        );
    rrdbnet_state_dict[ "conv_up2.weight" ] = (
        pretrained_state_dict[ "model.6.weight" ]
        );
    rrdbnet_state_dict[ "conv_up2.bias" ] = (
        pretrained_state_dict[ "model.6.bias" ]
        );
    rrdbnet_state_dict[ "conv_hr.weight" ] = (
        pretrained_state_dict[ "model.8.weight" ]
        );
    rrdbnet_state_dict[ "conv_hr.bias" ] = (
        pretrained_state_dict[ "model.8.bias" ]
        );
    rrdbnet_state_dict[ "conv_last.weight" ] = (
        pretrained_state_dict[ "model.10.weight" ]
        );
    rrdbnet_state_dict[ "conv_last.bias" ] = (
        pretrained_state_dict[ "model.10.bias" ]
        );

    return rrdbnet_state_dict;

# ~~

def get_state_dict_from_model_checkpoint(
    model_weights_file_path: str
    ) -> dict[ str, Any ]:

    checkpoint = (
        torch.load(
            model_weights_file_path,
            map_location=torch.device( "cpu" )
            )
        );

    if isinstance( checkpoint, dict ):

        if "params_ema" in checkpoint:

            state_dict = checkpoint[ "params_ema" ];

        elif "params" in checkpoint:

            state_dict = checkpoint[ "params" ];

        else:

            state_dict = checkpoint;

    else:

        state_dict = checkpoint;

    if is_old_arch_state_dict( state_dict ):

        state_dict = (
            convert_old_arch_state_dict_to_rrdbnet_state_dict(
                state_dict
                )
            );

    return state_dict;

# ~~

def get_model_block_count(
    model_name: str
    ) -> int:

    return MODEL_BLOCK_COUNT.get( model_name, DEFAULT_MODEL_BLOCK_COUNT );

# ~~

def get_model(
    model_name: str
    ) -> torch.nn.Module:

    if model_name in SRVGG_MODEL_NAME_SET:

        return (
            SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=DEFAULT_SRVGG_CONV_COUNT,
                upscale=MODEL_SCALE,
                act_type="prelu"
                )
            );

    return (
        RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=get_model_block_count( model_name ),
            num_grow_ch=32,
            scale=MODEL_SCALE
            )
        );

# ~~

def is_rocm_pytorch(
    ) -> bool:

    return getattr( torch.version, "hip", None ) is not None;

# ~~

def is_nvidia_cuda_available(
    ) -> bool:

    return torch.cuda.is_available() and not is_rocm_pytorch();

# ~~

def is_amd_rocm_available(
    ) -> bool:

    return torch.cuda.is_available() and is_rocm_pytorch();

# ~~

def resolve_compute_backend(
    requested_compute_backend: str | None
    ) -> str:

    nvidia_cuda_is_available = is_nvidia_cuda_available();
    amd_rocm_is_available = is_amd_rocm_available();

    if requested_compute_backend == "cpu":

        return "cpu";

    if requested_compute_backend == "cuda":

        if not nvidia_cuda_is_available:

            print( "CUDA was requested but is not available.", file=sys.stderr );
            sys.exit( 1 );

        return "cuda";

    if requested_compute_backend == "rocm":

        if not amd_rocm_is_available:

            print( "ROCm was requested but is not available.", file=sys.stderr );
            sys.exit( 1 );

        return "rocm";

    if nvidia_cuda_is_available:

        return "cuda";

    if amd_rocm_is_available:

        return "rocm";

    return "cpu";

# ~~

def get_upsampler(
    model_name: str,
    model_weights_file_path: str,
    tile_size: int,
    compute_backend: str
    ) -> SharpRealESRGANer:

    if compute_backend == "cpu":

        print( "Using CPU (slow).", file=sys.stderr );
        device = torch.device( "cpu" );
        half = False;

    elif compute_backend == "cuda":

        print( "Using CUDA.", file=sys.stderr );
        device = torch.device( "cuda" );
        half = True;

    else:

        print( "Using ROCm.", file=sys.stderr );
        device = torch.device( "cuda" );
        half = True;

    return (
        SharpRealESRGANer(
            scale=MODEL_SCALE,
            model_path=model_weights_file_path,
            model=get_model( model_name ),
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=half,
            device=device
            )
        );

# ~~

def input_image_meets_minimum_dimensions(
    input_width: int,
    input_height: int,
    minimum_width: int,
    minimum_height: int
    ) -> bool:

    return (
        ( minimum_width == 0 or input_width >= minimum_width )
        and ( minimum_height == 0 or input_height >= minimum_height )
        );

# ~~

def get_minimum_scale(
    input_width: int,
    input_height: int,
    minimum_width: int,
    minimum_height: int
    ) -> float:

    minimum_scale = 1.0;

    if minimum_width > 0:

        minimum_scale = max( minimum_scale, minimum_width / input_width );

    if minimum_height > 0:

        minimum_scale = max( minimum_scale, minimum_height / input_height );

    return minimum_scale;

# ~~

def apply_maximum_dimension_limits(
    output_width: int,
    output_height: int,
    maximum_width: int,
    maximum_height: int
    ) -> tuple[ int, int ]:

    if maximum_width == 0 and maximum_height == 0:

        return output_width, output_height;

    aspect_ratio = output_width / output_height;

    if maximum_width > 0 and output_width > maximum_width:

        output_width = maximum_width;
        output_height = int( round( output_width / aspect_ratio ) );

    if maximum_height > 0 and output_height > maximum_height:

        output_height = maximum_height;
        output_width = int( round( output_height * aspect_ratio ) );

    return output_width, output_height;

# ~~

def needs_upscaling(
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int
    ) -> bool:

    return (
        output_width > input_width
        or output_height > input_height
        );

# ~~

def get_output_dimensions(
    input_width: int,
    input_height: int,
    maximum_ratio: float,
    minimum_width: int,
    minimum_height: int,
    maximum_width: int,
    maximum_height: int,
    maximum_upscaled_width: int,
    maximum_upscaled_height: int
    ) -> tuple[ int, int ]:

    has_minimum_constraint = minimum_width > 0 or minimum_height > 0;

    if has_minimum_constraint:

        if input_image_meets_minimum_dimensions(
            input_width,
            input_height,
            minimum_width,
            minimum_height
            ):

            output_width = input_width;
            output_height = input_height;

        else:

            output_scale = get_minimum_scale(
                input_width,
                input_height,
                minimum_width,
                minimum_height
                );

            if output_scale > maximum_ratio:

                output_scale = maximum_ratio;

            output_width = int( round( input_width * output_scale ) );
            output_height = int( round( input_height * output_scale ) );

    else:

        output_width = int( round( input_width * MODEL_SCALE ) );
        output_height = int( round( input_height * MODEL_SCALE ) );

        if maximum_ratio < MODEL_SCALE:

            output_width = int( round( input_width * maximum_ratio ) );
            output_height = int( round( input_height * maximum_ratio ) );

    if needs_upscaling(
        input_width,
        input_height,
        output_width,
        output_height
        ):

        output_width, output_height = apply_maximum_dimension_limits(
            output_width,
            output_height,
            maximum_upscaled_width,
            maximum_upscaled_height
            );

    return apply_maximum_dimension_limits(
        output_width,
        output_height,
        maximum_width,
        maximum_height
        );

# ~~

def get_resized_image(
    input_image: np.ndarray,
    output_width: int,
    output_height: int
    ) -> np.ndarray:

    return cv2.resize(
        input_image,
        ( output_width, output_height ),
        interpolation=cv2.INTER_LANCZOS4
        );

# ~~

_is_avif_opener_registered = False;

def register_avif_opener(
    ) -> None:

    global _is_avif_opener_registered;

    if _is_avif_opener_registered:

        return;

    try:

        from pillow_avif import register_avif_opener as register_opener;

    except ImportError as import_error:

        print(
            f"Missing AVIF dependency: {import_error}",
            file=sys.stderr
            );
        print(
            "Install with: run install_packages_cuda.bat, install_packages_rocm.bat, or install_packages_cpu.bat",
            file=sys.stderr
            );
        sys.exit( 1 );

    register_opener();
    _is_avif_opener_registered = True;

# ~~

def get_bgr_image_and_alpha_channel_from_pil_image(
    pil_image
    ) -> tuple[ np.ndarray, np.ndarray | None ]:

    if pil_image.mode == "RGBA":

        rgba_image = np.array( pil_image );
        bgr_image = cv2.cvtColor( rgba_image, cv2.COLOR_RGBA2BGR );
        alpha_channel = rgba_image[ :, :, 3 ];

        return bgr_image, alpha_channel;

    if pil_image.mode == "LA":

        rgba_image = np.array( pil_image.convert( "RGBA" ) );
        bgr_image = cv2.cvtColor( rgba_image, cv2.COLOR_RGBA2BGR );
        alpha_channel = rgba_image[ :, :, 3 ];

        return bgr_image, alpha_channel;

    if pil_image.mode == "L":

        gray_image = np.array( pil_image );

        return cv2.cvtColor( gray_image, cv2.COLOR_GRAY2BGR ), None;

    rgb_image = np.array( pil_image.convert( "RGB" ) );

    return cv2.cvtColor( rgb_image, cv2.COLOR_RGB2BGR ), None;

# ~~

def read_avif_input_image(
    input_image_file_path: str
    ) -> tuple[ np.ndarray | None, np.ndarray | None ]:

    register_avif_opener();

    try:

        from PIL import Image;

    except ImportError as import_error:

        print(
            f"Missing AVIF dependency: {import_error}",
            file=sys.stderr
            );
        print(
            "Install with: run install_packages_cuda.bat, install_packages_rocm.bat, or install_packages_cpu.bat",
            file=sys.stderr
            );
        sys.exit( 1 );

    try:

        with Image.open( input_image_file_path ) as pil_image:

            return get_bgr_image_and_alpha_channel_from_pil_image( pil_image );

    except OSError:

        return None, None;

# ~~

def read_input_image(
    input_image_file_path: str
    ) -> tuple[ np.ndarray | None, np.ndarray | None ]:

    input_extension = os.path.splitext( input_image_file_path )[ 1 ].lower();

    if input_extension == ".avif":

        return read_avif_input_image( input_image_file_path );

    input_image = cv2.imread( input_image_file_path, cv2.IMREAD_UNCHANGED );

    if input_image is None:

        return None, None;

    if input_image.ndim == 2:

        return cv2.cvtColor( input_image, cv2.COLOR_GRAY2BGR ), None;

    channel_count = input_image.shape[ 2 ];

    if channel_count == 4:

        return input_image[ :, :, :3 ], input_image[ :, :, 3 ];

    if channel_count == 3:

        return input_image, None;

    return input_image[ :, :, :3 ], None;

# ~~

def get_processed_alpha_channel(
    alpha_channel: np.ndarray,
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int,
    needs_upscaling: bool
    ) -> np.ndarray:

    if not needs_upscaling:

        return get_resized_image(
            alpha_channel,
            output_width,
            output_height
            );

    intermediate_width = int( round( input_width * MODEL_SCALE ) );
    intermediate_height = int( round( input_height * MODEL_SCALE ) );

    output_alpha_channel = get_resized_image(
        alpha_channel,
        intermediate_width,
        intermediate_height
        );

    if (
        output_width != intermediate_width
        or output_height != intermediate_height
        ):

        output_alpha_channel = get_resized_image(
            output_alpha_channel,
            output_width,
            output_height
            );

    return output_alpha_channel;

# ~~

def merge_bgr_and_alpha_channel(
    bgr_image: np.ndarray,
    alpha_channel: np.ndarray
    ) -> np.ndarray:

    return cv2.merge(
        (
            bgr_image[ :, :, 0 ],
            bgr_image[ :, :, 1 ],
            bgr_image[ :, :, 2 ],
            alpha_channel
            )
        );

# ~~

def composite_image_over_alpha_color(
    input_image: np.ndarray,
    alpha_channel: np.ndarray,
    alpha_color_bgr: tuple[ int, int, int ]
    ) -> np.ndarray:

    alpha_factor = alpha_channel.astype( np.float32 ) / 255.0;
    alpha_factor_3d = alpha_factor[ :, :, np.newaxis ];
    alpha_color = np.array( alpha_color_bgr, dtype=np.float32 );

    return (
        np.clip(
            alpha_factor_3d * input_image.astype( np.float32 )
            + ( 1.0 - alpha_factor_3d ) * alpha_color,
            0,
            255
            )
        ).astype( np.uint8 );

# ~~

def finalize_output_image_with_alpha(
    bgr_image: np.ndarray,
    alpha_channel: np.ndarray,
    alpha_mode: str,
    alpha_color_bgr: tuple[ int, int, int ],
    output_extension: str
    ) -> np.ndarray:

    is_jpeg_output = output_extension in ( ".jpg", ".jpeg" );

    if is_jpeg_output or alpha_mode == "remove":

        return composite_image_over_alpha_color(
            bgr_image,
            alpha_channel,
            alpha_color_bgr
            );

    return merge_bgr_and_alpha_channel( bgr_image, alpha_channel );

# ~~

def is_loadable_image_extension(
    file_extension: str
    ) -> bool:

    return file_extension.lower() in SUPPORTED_IMAGE_EXTENSION_SET;

# ~~

def is_loadable_image_file_path(
    file_path: str
    ) -> bool:

    return (
        os.path.isfile( file_path )
        and is_loadable_image_extension(
            os.path.splitext( file_path )[ 1 ]
            )
        );

# ~~

def get_file_path_set_for_filter_list(
    input_image_folder_path: str,
    file_path_filter_list: list[ str ]
    ) -> set[ str ]:

    input_image_file_path_set: set[ str ] = set();

    for file_path_filter in file_path_filter_list:

        file_path_filter = file_path_filter.strip();

        if file_path_filter == "":

            continue;

        glob_pattern = input_image_folder_path + file_path_filter;

        input_image_file_path_set.update(
            normalize_file_path( input_image_file_path )
            for input_image_file_path in glob.glob( glob_pattern, recursive=True )
            if is_loadable_image_file_path( input_image_file_path )
            );

    return input_image_file_path_set;

# ~~

def get_all_loadable_input_image_file_path_set(
    input_image_folder_path: str
    ) -> set[ str ]:

    input_image_file_path_set: set[ str ] = set();

    for supported_image_extension in SUPPORTED_IMAGE_EXTENSION_SET:

        glob_pattern = (
            input_image_folder_path + "**/*" + supported_image_extension
            );

        input_image_file_path_set.update(
            normalize_file_path( input_image_file_path )
            for input_image_file_path in glob.glob( glob_pattern, recursive=True )
            if is_loadable_image_file_path( input_image_file_path )
            );

    return input_image_file_path_set;

# ~~

def get_input_image_file_path_list(
    input_image_folder_path: str,
    input_image_file_path_inclusion_filter_list: list[ str ],
    input_image_file_path_exclusion_filter_list: list[ str ]
    ) -> list[ str ]:

    if input_image_file_path_inclusion_filter_list:

        input_image_file_path_set = (
            get_file_path_set_for_filter_list(
                input_image_folder_path,
                input_image_file_path_inclusion_filter_list
                )
            );

    else:

        input_image_file_path_set = (
            get_all_loadable_input_image_file_path_set(
                input_image_folder_path
                )
            );

    if input_image_file_path_exclusion_filter_list:

        excluded_input_image_file_path_set = (
            get_file_path_set_for_filter_list(
                input_image_folder_path,
                input_image_file_path_exclusion_filter_list
                )
            );

        input_image_file_path_set -= excluded_input_image_file_path_set;

    return sorted( input_image_file_path_set );

# ~~

def get_template_variable_by_name_dictionary(
    input_image_folder_path: str,
    input_image_file_path: str,
    maximum_ratio: float,
    output_width: int,
    output_height: int
    ) -> dict[ str, str ]:

    input_image_file_path = normalize_file_path( input_image_file_path );
    input_relative_file_path = (
        get_input_relative_file_path(
            input_image_folder_path,
            input_image_file_path
            )
        );
    input_relative_directory_path = os.path.dirname( input_relative_file_path );

    if input_relative_directory_path == "":

        input_relative_directory_path = "";

    else:

        input_relative_directory_path = input_relative_directory_path.replace( "\\", "/" ) + "/";

    input_file_name = os.path.basename( input_image_file_path );
    input_file_stem, input_file_extension = os.path.splitext( input_file_name );

    return (
        {
            "f": input_relative_file_path,
            "d": input_relative_directory_path,
            "n": input_file_name,
            "s": input_file_stem,
            "e": input_file_extension.lstrip( "." ),
            "r": str( maximum_ratio ).rstrip( "0" ).rstrip( "." ),
            "w": str( output_width ),
            "h": str( output_height )
        }
        );

# ~~

def format_output_image_file_name(
    output_image_file_name_template: str,
    template_variable_by_name_dictionary: dict[ str, str ]
    ) -> str:

    output_image_file_name = output_image_file_name_template;

    for variable_name, variable_value in template_variable_by_name_dictionary.items():

        output_image_file_name = (
            output_image_file_name.replace(
                "{" + variable_name + "}",
                variable_value
                )
            );

    return output_image_file_name;

# ~~

def write_avif_output_image(
    output_image_file_path: str,
    output_image: np.ndarray,
    avif_compression: int
    ) -> None:

    register_avif_opener();

    try:

        from PIL import Image;

    except ImportError as import_error:

        print(
            f"Missing AVIF dependency: {import_error}",
            file=sys.stderr
            );
        print(
            "Install with: run install_packages_cuda.bat, install_packages_rocm.bat, or install_packages_cpu.bat",
            file=sys.stderr
            );
        sys.exit( 1 );

    channel_count = (
        1
        if output_image.ndim == 2
        else output_image.shape[ 2 ]
        );

    if channel_count == 4:

        rgba_image = cv2.cvtColor( output_image, cv2.COLOR_BGRA2RGBA );
        pil_image = Image.fromarray( rgba_image );

    elif channel_count == 3:

        rgb_image = cv2.cvtColor( output_image, cv2.COLOR_BGR2RGB );
        pil_image = Image.fromarray( rgb_image );

    else:

        pil_image = Image.fromarray( output_image );

    if avif_compression >= 100:

        pil_image.save(
            output_image_file_path,
            format="AVIF",
            quality=avif_compression,
            subsampling="4:4:4",
            range="full",
            speed=0
            );

    else:

        pil_image.save(
            output_image_file_path,
            format="AVIF",
            quality=avif_compression
            );

# ~~

def write_output_image(
    output_image_file_path: str,
    output_image: np.ndarray,
    avif_compression: int,
    jpeg_compression: int,
    webp_compression: int
    ) -> None:

    output_image_file_path = normalize_file_path( output_image_file_path );
    output_image_file_folder_path = (
        get_output_image_file_folder_path( output_image_file_path )
        );

    if output_image_file_folder_path != "":

        os.makedirs( output_image_file_folder_path.rstrip( "/" ), exist_ok=True );

    output_extension = os.path.splitext( output_image_file_path )[ 1 ].lower();

    if output_extension == ".avif":

        write_avif_output_image(
            output_image_file_path,
            output_image,
            avif_compression
            );
        is_write_successful = os.path.isfile( output_image_file_path );

    elif output_extension in ( ".jpg", ".jpeg" ):

        is_write_successful = cv2.imwrite(
            output_image_file_path,
            output_image,
            [ cv2.IMWRITE_JPEG_QUALITY, jpeg_compression ]
            );

    elif output_extension == ".webp":

        is_write_successful = cv2.imwrite(
            output_image_file_path,
            output_image,
            [ cv2.IMWRITE_WEBP_QUALITY, webp_compression ]
            );

    else:

        is_write_successful = cv2.imwrite( output_image_file_path, output_image );

    if not is_write_successful:

        raise OSError(
            f"Failed to write image: {output_image_file_path}"
            );

# ~~

def upscale_images(
    input_image_folder_path: str,
    input_image_file_path_inclusion_filter_list: list[ str ],
    input_image_file_path_exclusion_filter_list: list[ str ],
    output_image_folder_path: str,
    output_image_file_path_template: str,
    upsampler: SharpRealESRGANer | None,
    maximum_ratio: float,
    avif_compression: int,
    jpeg_compression: int,
    webp_compression: int,
    minimum_width: int,
    minimum_height: int,
    maximum_width: int,
    maximum_height: int,
    maximum_upscaled_width: int,
    maximum_upscaled_height: int,
    alpha_mode: str,
    alpha_color_bgr: tuple[ int, int, int ],
    is_skip_enabled: bool
    ) -> None:

    os.makedirs( output_image_folder_path.rstrip( "/" ), exist_ok=True );

    input_image_file_path_list = (
        get_input_image_file_path_list(
            input_image_folder_path,
            input_image_file_path_inclusion_filter_list,
            input_image_file_path_exclusion_filter_list
            )
        );

    if not input_image_file_path_list:

        if input_image_file_path_inclusion_filter_list:

            inclusion_message = (
                ", ".join( input_image_file_path_inclusion_filter_list )
                );

        else:

            inclusion_message = "all loadable images";

        exclusion_message = "";

        if input_image_file_path_exclusion_filter_list:

            exclusion_message = (
                f" (exclude: {', '.join( input_image_file_path_exclusion_filter_list )})"
                );

        print(
            "No matching images found: "
            f"{input_image_folder_path}"
            f"{inclusion_message}"
            f"{exclusion_message}",
            file=sys.stderr
            );
        sys.exit( 1 );

    compute_device = upsampler.device if upsampler is not None else None;

    for input_image_file_path in input_image_file_path_list:

        print( f"Reading {input_image_file_path}" );

        input_image, input_alpha_channel = read_input_image( input_image_file_path );

        if input_image is None:

            print(
                f"Failed to read image: {input_image_file_path}",
                file=sys.stderr
                );
            continue;

        input_height, input_width = input_image.shape[ :2 ];

        output_width, output_height = (
            get_output_dimensions(
                input_width,
                input_height,
                maximum_ratio,
                minimum_width,
                minimum_height,
                maximum_width,
                maximum_height,
                maximum_upscaled_width,
                maximum_upscaled_height
                )
            );

        template_variable_by_name_dictionary = (
            get_template_variable_by_name_dictionary(
                input_image_folder_path,
                input_image_file_path,
                maximum_ratio,
                output_width,
                output_height
                )
            );

        output_image_file_name = (
            format_output_image_file_name(
                output_image_file_path_template,
                template_variable_by_name_dictionary
                )
            );

        output_image_file_path = (
            output_image_folder_path + output_image_file_name
            );

        if is_skip_enabled and os.path.isfile( output_image_file_path ):

            try:

                input_modified_time = os.path.getmtime( input_image_file_path );
                output_modified_time = os.path.getmtime( output_image_file_path );

            except OSError:

                input_modified_time = 0;
                output_modified_time = -1;

            if output_modified_time >= input_modified_time:

                print( f"Skipping {output_image_file_path}" );
                continue;

        output_extension = os.path.splitext( output_image_file_path )[ 1 ].lower();

        if not is_loadable_image_extension( output_extension ):

            print(
                "Output image file path must have a "
                ".avif, .jpg, .jpeg, .png, or .webp extension: "
                f"{output_image_file_path}",
                file=sys.stderr
                );
            sys.exit( 1 );

        is_upscaling_needed = (
            needs_upscaling(
                input_width,
                input_height,
                output_width,
                output_height
                )
            );

        is_realesrnet_alpha_upscale_enabled = (
            input_alpha_channel is not None
            and alpha_mode == "realesrnet"
            and is_upscaling_needed
            );

        if not is_upscaling_needed:

            output_image = (
                get_resized_image(
                    input_image,
                    output_width,
                    output_height
                    )
                );

        elif is_realesrnet_alpha_upscale_enabled:

            input_bgra_image = (
                merge_bgr_and_alpha_channel(
                    input_image,
                    input_alpha_channel
                    )
                );

            try:

                output_image, _unused_image_mode = upsampler.enhance(
                    input_bgra_image,
                    outscale=MODEL_SCALE,
                    alpha_upsampler="realesrgan"
                    );

            except RuntimeError as runtime_error:

                print( f"Error: {runtime_error}", file=sys.stderr );
                print(
                    "Try again with tile size 400 (or a smaller value).",
                    file=sys.stderr
                    );
                sys.exit( 1 );

            if (
                output_width != input_width * MODEL_SCALE
                or output_height != input_height * MODEL_SCALE
                ):

                output_image = (
                    get_resized_image(
                        output_image,
                        output_width,
                        output_height
                        )
                    );

        else:

            try:

                output_image, _unused_image_mode = upsampler.enhance(
                    input_image,
                    outscale=MODEL_SCALE
                    );

            except RuntimeError as runtime_error:

                print( f"Error: {runtime_error}", file=sys.stderr );
                print(
                    "Try again with tile size 400 (or a smaller value).",
                    file=sys.stderr
                    );
                sys.exit( 1 );

            if (
                output_width != input_width * MODEL_SCALE
                or output_height != input_height * MODEL_SCALE
                ):

                output_image = (
                    get_resized_image(
                        output_image,
                        output_width,
                        output_height
                        )
                    );

        if input_alpha_channel is not None:

            if is_realesrnet_alpha_upscale_enabled:

                output_bgr_image = output_image[ :, :, :3 ];
                output_alpha_channel = output_image[ :, :, 3 ];

            else:

                output_bgr_image = output_image;
                output_alpha_channel = (
                    get_processed_alpha_channel(
                        input_alpha_channel,
                        input_width,
                        input_height,
                        output_width,
                        output_height,
                        is_upscaling_needed
                        )
                    );

            output_image = (
                finalize_output_image_with_alpha(
                    output_bgr_image,
                    output_alpha_channel,
                    alpha_mode,
                    alpha_color_bgr,
                    output_extension
                    )
                );

        print( f"Writing {output_image_file_path}" );

        try:

            write_output_image(
                output_image_file_path,
                output_image,
                avif_compression,
                jpeg_compression,
                webp_compression
                );

        except Exception as write_error:

            print( f"Error: {write_error}", file=sys.stderr );

            if os.path.isfile( output_image_file_path ):

                os.remove( output_image_file_path );

            sys.exit( 1 );

        if compute_device is not None and compute_device.type == "cuda":

            torch.cuda.synchronize( compute_device );

# ~~

def main(
    ) -> None:

    command_line_arguments = parse_arguments();

    input_image_folder_path = (
        normalize_folder_path(
            command_line_arguments.input_image_folder_path
            )
        );
    output_image_folder_path = (
        normalize_folder_path(
            command_line_arguments.output_image_folder_path
            )
        );

    validate_input_image_folder_path( input_image_folder_path );
    validate_maximum_ratio( command_line_arguments.max_ratio );
    validate_minimum_dimensions(
        command_line_arguments.min_width,
        command_line_arguments.min_height
        );
    validate_maximum_upscaled_dimensions(
        command_line_arguments.max_upscaled_width,
        command_line_arguments.max_upscaled_height
        );
    validate_output_image_file_name_template(
        command_line_arguments.template
        );
    alpha_color_bgr = parse_alpha_color( command_line_arguments.alpha_color );
    avif_compression, jpeg_compression, webp_compression = (
        resolve_compression_settings( command_line_arguments )
        );

    upsampler = None;

    if command_line_arguments.max_ratio > 1:

        compute_backend = (
            resolve_compute_backend( command_line_arguments.compute_backend )
            );

        upsampler = (
            get_upsampler(
                command_line_arguments.model,
                get_model_weights_file_path( command_line_arguments.model ),
                tile_size=command_line_arguments.tile_size,
                compute_backend=compute_backend
                )
            );

    upscale_images(
        input_image_folder_path,
        command_line_arguments.include or [],
        command_line_arguments.exclude or [],
        output_image_folder_path,
        command_line_arguments.template,
        upsampler,
        maximum_ratio=command_line_arguments.max_ratio,
        avif_compression=avif_compression,
        jpeg_compression=jpeg_compression,
        webp_compression=webp_compression,
        minimum_width=command_line_arguments.min_width,
        minimum_height=command_line_arguments.min_height,
        maximum_width=command_line_arguments.max_width,
        maximum_height=command_line_arguments.max_height,
        maximum_upscaled_width=command_line_arguments.max_upscaled_width,
        maximum_upscaled_height=command_line_arguments.max_upscaled_height,
        alpha_mode=command_line_arguments.alpha_mode,
        alpha_color_bgr=alpha_color_bgr,
        is_skip_enabled=command_line_arguments.skip
        );

# -- STATEMENTS

if __name__ == "__main__":

    main();
