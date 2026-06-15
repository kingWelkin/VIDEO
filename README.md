# Scathach RIFE Wallpaper

Dense A-B-A RIFE interpolation workflow for a looping Wallpaper Engine video wallpaper.

## Output

- `wallpaper_engine_scathach/scathach_two_source_rife_interpolated_2160p60_6s.mp4`
- `wallpaper_engine_scathach/project.json`
- `wallpaper_engine_scathach/preview.jpg`

The delivered video is 3840x2160, 60 fps, 6 seconds, 360 frames.

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

Run tests:

```powershell
& 'C:\Users\Donghao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest .\test_two_source_rife_interpolated.py
```
