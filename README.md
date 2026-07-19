![](https://github.com/senselogic/SHARP/blob/master/LOGO/sharp.png)

# Sharp

GPU-accelerated AI image upscaler.

## Command line

```
sharp <input image folder path> <output image folder path> [<options>]
```

or

```
sharp_uv <input image folder path> <output image folder path> [<options>]
```

## Template variables

```
{f} : input relative file path
{d} : input relative directory path
{n} : input file name
{s} : input file stem
{e} : input file extension
{r} : output scaling ratio
{w} : output width
{h} : output height
```

## Options

```
--include <input_image_file_path_inclusion_filter>
--exclude <input_image_file_path_exclusion_filter>
--template <output_image_file_path_template={d}{s}.png>
--model <model_name=remacri>
--min-width <minimum_width=0>
--min-height <minimum_height=0>
--max-ratio <maximum_ratio=4>
--max-width <maximum_width=0>
--max-height <maximum_height=0>
--max-upscaled-width <maximum_upscaled_width=0>
--max-upscaled-height <maximum_upscaled_height=0>
--tile-size <tile_size=400|0>
--compression <compression>
--avif-compression <avif_compression=85>
--jpeg-compression <jpeg_compression=85>
--webp-compression <webp_compression=85>
--alpha-mode <alpha_mode=lanczos|realesrnet|remove>
--alpha-color <alpha_color=FFFFFF>
--skip
```

## Models

- `realanime` — anime and manga
- `realdigital` — digital art
- `realesrgan` — sharp textures
- `realesrnet` — smooth output with minimal invented detail
- `remacri` — strong on skin, faces, and fine textures
- `ultramix` — balanced detail and smoothness
- `ultrasharp` — aggressive detail recovery

## Samples

```
sharp "IN/" "OUT/"
```

```
sharp "IN/" "OUT/" --model remacri
```

```
sharp "IN/" "OUT/" --include "**/*.jpg" --include "**/*.png" --template "{d}{s}.jpg" --skip
```

```
sharp "IN/" "OUT/" --include "**/*.jpg" --include "**/*.png" --template "{d}{s}_{r}x.jpg" --max-ratio 3 --skip
```

```
sharp "IN/" "OUT/" --include "**/*.jpg" --include "**/*.png" --template "{d}{s}_{w}x{h}.png" --max-width 3840 --compression 85 --skip
```

## Install

Run `install_packages.bat` or `install_uv_packages.bat`.

## Dependencies

- Python 3.10.11
- CUDA 12.4

## Limitations

- Only reads and writes AVIF, JPEG, PNG, and WebP images.
- Output resolution is limited to 4× the original resolution.

## Version

0.1

## Author

Eric Pelzer (ecstatic.coder@gmail.com).

## License

This project is licensed under the GNU General Public License version 3.

See the [LICENSE.md](LICENSE.md) file for details.
