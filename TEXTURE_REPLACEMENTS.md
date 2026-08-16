# RPCS3 外部高清纹理替换（V1）

这个实现参考 PCSX2 与 PPSSPP 的成熟做法：在模拟器已经识别并解码游戏纹理之后、创建宿主 GPU 纹理之前进行匹配与替换。它不修改 `PSARC`、游戏文件或 PS3 客体内存，因此高清纹理只占用电脑显存，不会再次触碰《战神：升天》的游戏内档案尺寸和显存预算。

参考实现：

- [PCSX2 `GSTextureReplacements`](https://github.com/PCSX2/pcsx2/blob/master/pcsx2/GS/Renderers/HW/GSTextureReplacements.cpp)
- [PPSSPP 纹理替换说明](https://dev.ppsspp.org/docs/reference/use-texture-replacement/)
- [PPSSPP `TextureReplacer`](https://github.com/hrydgard/ppsspp/blob/master/GPU/Common/TextureReplacer.cpp)

## 快速使用

1. 在游戏的自定义配置中选择 Vulkan，启用 `Dump Replaceable Textures`。
2. 运行需要采集的场景。PNG 会写入：
   `config/textures/<TITLE_ID>/dumps/`
3. 关闭导出开关。保留原文件名，将处理后的 PNG 放到：
   `config/textures/<TITLE_ID>/replacements/`
4. 启用 `Load Texture Replacements`，重新启动游戏。

以《战神：升天》BCAS25016 为例，目录是：

```text
config/textures/BCAS25016/
  dumps/
  replacements/
```

文件名中的 SHA-1 同时覆盖原始纹理内容、格式、尺寸、mipmap 和通道映射；同名 PNG 才会替换对应纹理。替换文件必须保持相同比例，并且是原图的 1–8 倍整数缩放。无文件、解码失败、比例错误、尺寸过大或当前 GPU 不支持时，RPCS3 会继续使用原纹理。

## V1 范围

- Vulkan 加载路径；PNG/RGBA；二维静态采样纹理。
- 当前候选格式：A8R8G8B8、D8R8G8B8、DXT1、DXT3、DXT5。
- 有意排除 render target、深度、立方体、三维纹理，以及大多数线性动态纹理，以免把视频帧、HUD 合成面或不断变化的缓冲误当材质。
- V1 只接受一张基础 PNG；自定义 mip 链、纹理别名/规则文件、热重载和 OpenGL 支持留给后续版本。
- 修改纹理包后应重新启动游戏；缓存中的纹理不会在运行中自动刷新。

## 法线贴图注意事项

法线贴图不是普通彩色照片。处理时必须保留通道、alpha 和 DirectX/OpenGL 法线方向，放大后应重新归一化法线。通用照片增强模型可能制造油膜、彩虹高光或角度相关的光栅感；这类结果不是 RPCS3 采样错误，应该从贴图处理流程修正。

## 为什么不继续增大 PSARC

重新打包会让高清资源进入游戏自身的 32 位地址空间、分配器、流式加载和显存管理契约。外部替换把身份匹配留在原始纹理上，把高清副本放在宿主 GPU 侧：游戏仍认为自己加载的是原尺寸资源，模拟器只在最后上传阶段换成高清图。这正是此功能相对“巨型高清 PSARC”的核心价值。
