# Scathach RIFE Wallpaper

Dense A-B-A RIFE interpolation workflow for a looping Wallpaper Engine video wallpaper.

## Output

- `wallpaper_engine_scathach/scathach_two_source_rife_interpolated_2160p60_4s_v6_highpass_deghost.mp4`
- `wallpaper_engine_scathach/scathach_two_source_rife_interpolated_2160p60_4s_v5_deghost.mp4`
- `wallpaper_engine_scathach/scathach_two_source_rife_interpolated_2160p60_4s_v4_plate.mp4`
- `wallpaper_engine_scathach/scathach_two_source_rife_interpolated_2160p60_4s_v2.mp4`
- `wallpaper_engine_scathach/scathach_two_source_rife_interpolated_2160p60_4s_v3.mp4`
- `wallpaper_engine_scathach/scathach_two_source_rife_interpolated_2160p60_6s.mp4`
- `wallpaper_engine_scathach/project.json`
- `wallpaper_engine_scathach/preview.jpg`

The current Wallpaper Engine project points to the v6 highpass deghost video: 3840x2160, 60 fps, 4 seconds, 240 frames.
The earlier accepted 6 second, v2, v3, v4, and v5 versions are kept in the same project folder.

## Rebuild

Place the two-image source package at:

```text
images_3840x2160.zip
```

The RIFE executable is expected at:

```text
D:/codex_video_tools/rife-ncnn-vulkan-20221029-windows/rife-ncnn-vulkan.exe
```

Run:

```powershell
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\make_scathach_wallpaper_two_source_rife_interpolated.py --mode all --overwrite-rife --overwrite-effects --overwrite-sources
```

Run v2:

```powershell
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\make_scathach_wallpaper_two_source_rife_interpolated_v2.py --mode all --overwrite-rife --overwrite-effects
```

Run v3:

```powershell
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\make_scathach_wallpaper_two_source_rife_interpolated_v3.py --mode all --overwrite-effects
```

Run v4:

```powershell
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\make_scathach_wallpaper_two_source_rife_interpolated_v4.py --mode all
```

Run v5:

```powershell
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\make_scathach_wallpaper_two_source_rife_interpolated_v5.py --mode all
```

Run v6:

```powershell
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\make_scathach_wallpaper_two_source_rife_interpolated_v6.py --mode all
```

Run tests:

```powershell
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest .\test_two_source_rife_interpolated.py
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest .\test_two_source_rife_interpolated_v2.py
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest .\test_two_source_rife_interpolated_v3.py
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest .\test_two_source_rife_interpolated_v4.py
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest .\test_two_source_rife_interpolated_v5.py
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest .\test_two_source_rife_interpolated_v6.py
```
