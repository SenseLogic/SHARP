call "..\sharp.bat" "IN/" "IN/" --include "anime/*.*" --include "architectural/*.*" --include "digital/*.*" --include "huge/*.*" --include "large/*.*" --template "{d}small/{s}.png" --max-width 960 --skip
call "..\sharp.bat" "IN/" "OUT/" --include "alpha/*.*" --template "{d}{s}.png" --skip
call "..\sharp.bat" "IN/" "OUT/" --include "anime/*.*" --template "{d}{s}.jpg" --model realanime --skip
call "..\sharp.bat" "IN/" "OUT/" --include "digital/*.*" --template "{d}{s}.jpg" --model realdigital --skip
call "..\sharp.bat" "IN/" "OUT/" --include "architectural/*.*" --include "**/small/*.*" --template "{d}{s}_bsrgan.jpg" --model bsrgan --skip
call "..\sharp.bat" "IN/" "OUT/" --include "architectural/*.*" --include "**/small/*.*" --template "{d}{s}_bsrnet.jpg" --model bsrnet --skip
call "..\sharp.bat" "IN/" "OUT/" --include "architectural/*.*" --include "**/small/*.*" --template "{d}{s}_highfidelity.jpg" --model highfidelity --skip
call "..\sharp.bat" "IN/" "OUT/" --include "architectural/*.*" --include "**/small/*.*" --template "{d}{s}_realanime.jpg" --model realanime --skip
call "..\sharp.bat" "IN/" "OUT/" --include "architectural/*.*" --include "**/small/*.*" --template "{d}{s}_realdigital.jpg" --model realdigital --skip
call "..\sharp.bat" "IN/" "OUT/" --include "architectural/*.*" --include "**/small/*.*" --template "{d}{s}_realesrgan.jpg" --model realesrgan --skip
call "..\sharp.bat" "IN/" "OUT/" --include "architectural/*.*" --include "**/small/*.*" --template "{d}{s}_realesrnet.jpg" --model realesrnet --skip
call "..\sharp.bat" "IN/" "OUT/" --include "architectural/*.*" --include "**/small/*.*" --template "{d}{s}_remacri.jpg" --model remacri --skip
call "..\sharp.bat" "IN/" "OUT/" --include "architectural/*.*" --include "**/small/*.*" --template "{d}{s}_ultramix.jpg" --model ultramix --skip
call "..\sharp.bat" "IN/" "OUT/" --include "architectural/*.*" --include "**/small/*.*" --template "{d}{s}_ultrasharp.jpg" --model ultrasharp --skip
call "..\sharp.bat" "IN/" "OUT/" --include "**/*.jpg" --include "**/*.png" --exclude "alpha/*.*" --exclude "anime/*.*" --exclude "architectural/*.*" --exclude "digital/*.*" --exclude "**/small/*.*" --template "{d}{s}.jpg" --min-width 3840 --max-upscaled-width 3840 --skip
