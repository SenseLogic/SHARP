call "..\sharp_uv.bat" "IN/" "IN/" --include "large/*.*" --template "small/{s}.png" --max-width 960 --skip
call "..\sharp_uv.bat" "IN/" "OUT/" --include "alpha/*.png" --template "{d}{s}.png" --skip
call "..\sharp_uv.bat" "IN/" "OUT/" --include "small/*.*" --template "{d}{s}_realesrgan.jpg" --model realesrgan --skip
call "..\sharp_uv.bat" "IN/" "OUT/" --include "small/*.*" --template "{d}{s}_realesrnet.jpg" --model realesrnet --skip
call "..\sharp_uv.bat" "IN/" "OUT/" --include "small/*.*" --template "{d}{s}_remacri.jpg" --model remacri --skip
call "..\sharp_uv.bat" "IN/" "OUT/" --include "small/*.*" --template "{d}{s}_ultramix.jpg" --model ultramix --skip
call "..\sharp_uv.bat" "IN/" "OUT/" --include "small/*.*" --template "{d}{s}_ultrasharp_uv.jpg" --model ultrasharp --skip
call "..\sharp_uv.bat" "IN/" "OUT/" --include "**/*.jpg" --include "**/*.png" --exclude "alpha/*.*" --exclude "small/*.*" --template "{d}{s}.jpg" --skip
