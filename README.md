# Scathach RIFE Wallpaper

Dense A-B-A RIFE interpolation workflow for a looping Wallpaper Engine video wallpaper.

## Output

- `wallpaper_engine_scathach/scathach_two_source_rife_interpolated_2160p60_4s_v2.mp4`
- `wallpaper_engine_scathach/scathach_two_source_rife_interpolated_2160p60_6s.mp4`
- `wallpaper_engine_scathach/project.json`
- `wallpaper_engine_scathach/preview.jpg`

The current Wallpaper Engine project points to the v2 video: 3840x2160, 60 fps, 4 seconds, 240 frames.
The earlier accepted 6 second version is kept in the same project folder.

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

Run tests:

```powershell
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest .\test_two_source_rife_interpolated.py
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest .\test_two_source_rife_interpolated_v2.py
```
